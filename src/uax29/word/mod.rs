pub(crate) mod properties;
pub(crate) mod transitions;

use crate::uax29::Action;
use properties::{
    ASCII_WORD_BREAK_PROP, WordBreakProperty, is_word_like_strict,
    lookup_word_break_property_from_dictionary,
};
use transitions::{State, TABLE, Transition};

/// For backwards compatibility, require caller to pass in options struct.
#[derive(Default, Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Options {}

/// For a given span, extracts info from the DFA state to provide useful information upstream, e.g.
/// whether the span was "word-like", ascii, etc
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct TokenProperties(u8);

impl TokenProperties {
    const WORD_LIKE_MASK: u8 = 0b0000_0001;
    const NON_ASCII_MASK: u8 = 0b0000_0010;
    const HAS_ASCII_UPPER_MASK: u8 = 0b0000_0100;

    pub(crate) const NON_ASCII: Self = Self(Self::NON_ASCII_MASK);
    pub(crate) const WORD_LIKE: Self = Self(Self::WORD_LIKE_MASK);
    pub(crate) const HAS_ASCII_UPPER: Self = Self(Self::HAS_ASCII_UPPER_MASK);

    // A token is "word-like" if it contains any char that is:
    // - ALetter, HebrewLetter, or Numeric (this is a fast-path from our DFA WordBreakProperty lookup)
    // - Ideographic or Extended_Pictographic (e.g. CJK chars, emoji)
    // - Other_Number general category (⑦, ², ¼)
    // - A character whose Script is something meaningful (e.g. belonging to a real writing system),
    //   as opposed to Script=Common/Inherited/Unknown (e.g. punctuation, symbols, emoji modifiers).
    pub fn is_word_like(&self) -> bool {
        self.0 & Self::WORD_LIKE_MASK != 0
    }

    // Stored disjunctively: a single non-ASCII char in the span sets this bit.
    // `is_ascii()` returns true when the bit is unset (vacuously true for the empty span).
    pub fn is_ascii(&self) -> bool {
        self.0 & Self::NON_ASCII_MASK == 0
    }

    // Stored disjunctively: a single ASCII uppercase byte (A–Z) in the span sets this bit.
    // `has_ascii_upper()` returns true when the bit is set (vacuously false for the empty span).
    pub fn has_ascii_upper(&self) -> bool {
        self.0 & Self::HAS_ASCII_UPPER_MASK != 0
    }
}

impl std::ops::BitOrAssign for TokenProperties {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A tokenizer that implements UAX #29 word boundary rules, using a deterministic finite automaton
/// (DFA) to efficiently determine word boundaries in Unicode text. Includes a number of fast-paths
/// for common cases, e.g. ASCII.
pub fn tokenize(
    text: &str,
    _options: Options,
    mut on_breakpoint: impl FnMut(usize, TokenProperties) -> bool,
) {
    if text.is_empty() {
        return;
    }
    let bytes = text.as_bytes();

    let mut state = State::StartOfText;
    let mut deferred_break_pos = None;
    let mut pos = 0;

    // WB4 says: X (Extend | Format | ZWJ)*	→	X
    // To avoid adding _many_ `_AfterZWJ` variant states, we'll cheat a little by keeping track
    // of this condition with a bool. More specifically, we need to conditionally break based on
    // whether the previous character was a ZWJ.
    //
    // Example:
    // 'a 🛑' -> break (ALetter -> Other)
    // 'a ZWJ 🛑' -> no break (WB4)
    let mut last_was_zwj = false;

    // Maintain properties of the current token, which are reset on each break and can be used by the caller
    // to more efficiently determine what type of token was just emitted, e.g. whether it's "word-like" or ascii.
    let mut token_props = TokenProperties::default();

    // Properties of chars consumed while in a deferred state. Held aside from `token_props`
    // because we don't yet know which token they belong to: if the deferred state resolves
    // via `DeferredBreak`, these chars start the *next* token (so their contribution must
    // not leak into the in-progress one); if it resolves via `NoBreak` exiting deferred,
    // they fold into the current token. Tracked by `deferred_break_pos.is_some()`.
    let mut deferred_props = TokenProperties::default();

    while pos < text.len() {
        // Fast path for ASCII, e.g. skip DFA all together when possible.
        // Roughly a ~2x speedup on English Wikipedia.
        if matches!(
            state,
            State::ALetter | State::Numeric | State::ExtendNumLet | State::HLetter
        ) {
            let scan_start = pos;
            let (end, word_like, has_upper) = scan_word_continue(bytes, pos);
            pos = end;
            if pos > scan_start {
                // `bytes[pos - 1]` is in bounds: `pos > scan_start`, so `pos >= 1`.
                state = fold_ascii_word(&mut token_props, word_like, has_upper, bytes[pos - 1]);
                last_was_zwj = false;
                continue;
            }
        }

        // Fast path for ASCII, e.g. avoid chars().next(), and lookup word property from table.
        // `char_props` is this char's contribution to the enclosing token's properties; it's
        // applied to `token_props` per-arm below, since `Action::Break` treats the breaking char
        // as the first char of the *next* token (the contribution lands there, not in the token
        // being emitted).
        let b = bytes[pos];
        let (c, prop, char_len, char_props) = if b < 0x80 {
            (
                b as char,
                ASCII_WORD_BREAK_PROP[b as usize],
                1usize,
                // Reuses `fold_ascii_word`'s mapping (discarding the state it returns), so this
                // branch and the fast-lane run can't disagree on a byte's word-like / uppercase
                // contribution.
                {
                    let mut p = TokenProperties::default();
                    fold_ascii_word(&mut p, is_ascii_alnum(b), b.is_ascii_uppercase(), b);
                    p
                },
            )
        } else {
            let c = text[pos..].chars().next().unwrap();
            let prop = lookup_word_break_property_from_dictionary(c);
            // Cheap path covers ALetter / HebrewLetter / Numeric. For everything else, fall back
            // to the strict per-char check (ExtPict / Ideographic / Script / OtherNumber).
            let mut char_props = TokenProperties::NON_ASCII;
            char_props |= WORD_BREAK_CONTRIB[prop as usize];
            if !char_props.is_word_like() && is_word_like_strict(c) {
                char_props |= TokenProperties::WORD_LIKE;
            }
            (c, prop, c.len_utf8(), char_props)
        };

        // Each iteration, we consult the transition table to determine the next state
        // and whether to emit a breakpoint.
        let Transition(next_state, action) = TABLE[state as usize][prop as usize];
        match action {
            Action::Break => {
                let boundary = pos;
                pos += char_len;
                if last_was_zwj {
                    last_was_zwj = false;
                    if WordBreakProperty::is_ext_pictographic(c) {
                        // Transparent: char joins the in-progress token instead of breaking.
                        token_props |= char_props;
                        continue;
                    }
                }
                last_was_zwj = prop == WordBreakProperty::ZWJ;
                state = next_state;
                if !on_breakpoint(boundary, std::mem::take(&mut token_props)) {
                    return;
                }
                // Breaking char starts the next token; apply its contribution after the take.
                token_props |= char_props;
                continue;
            }
            Action::NoBreak => {
                last_was_zwj = false;
                if next_state.is_deferred() {
                    if deferred_break_pos.is_none() {
                        deferred_break_pos = Some(pos);
                    }
                    deferred_props |= char_props;
                } else {
                    if deferred_break_pos.take().is_some() {
                        // Word resumed: deferred chars belong to the in-progress token.
                        token_props |= std::mem::take(&mut deferred_props);
                    }
                    token_props |= char_props;
                }
                state = next_state;
                pos += char_len;
            }
            Action::DeferredBreak => {
                last_was_zwj = false;
                let boundary = deferred_break_pos.take().unwrap();
                state = next_state;
                // Notably, we don't advance `pos` here; the current char is re-examined on the
                // next iteration and will accumulate its props then — don't apply char_props here.
                if !on_breakpoint(boundary, std::mem::take(&mut token_props)) {
                    return;
                }
                // Deferred chars start the next token.
                token_props |= std::mem::take(&mut deferred_props);
                continue;
            }
            Action::Transparent => {
                last_was_zwj = prop == WordBreakProperty::ZWJ;
                // State doesn't change, but we still consume the character.
                pos += char_len;
                if deferred_break_pos.is_some() {
                    deferred_props |= char_props;
                } else {
                    token_props |= char_props;
                }
            }
        }
    }

    // Deferred state at EOT - defer failed
    if state.is_deferred() {
        let breakpoint = deferred_break_pos.take().unwrap();
        if !on_breakpoint(breakpoint, std::mem::take(&mut token_props)) {
            return;
        }
        // Deferred chars become the trailing token.
        token_props |= std::mem::take(&mut deferred_props);
    }

    // WB2: Any ÷ eot — emit final segment
    _ = on_breakpoint(text.len(), token_props);
}

/// Scans the ASCII "word-continue" run `[a-zA-Z0-9_]` starting at `start`, returning the index of
/// the first byte that is *not* word-continue (or `bytes.len()`), whether the consumed run
/// contained at least one alphanumeric char (i.e. is "word-like" — a run of only `_` is not), and
/// whether it contained at least one ASCII uppercase byte (`[A-Z]`) — see
/// [`TokenProperties::has_ascii_upper`].
///
/// This is the tokenizer's hottest loop on Latin-script text, so it classifies bytes with pure
/// arithmetic (no per-byte table load: removes a load→load dependency that capped the original
/// loop at ~1 cycle/byte) and processes a `usize` word at a time via SWAR. The uppercase flag rides
/// along this same pass for free, so the caller never needs a second per-token scan.
#[inline]
fn scan_word_continue(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    let mut pos = start;
    let mut word_like = false;
    let mut has_upper = false;

    // SWAR fast lane: classify `WORD_BYTES` bytes per iteration. We only enter the word-at-a-time
    // path while a full word remains; the scalar tail below finishes the run.
    while pos + WORD_BYTES <= bytes.len() {
        // SAFETY: the `while` condition guarantees `pos + WORD_BYTES <= bytes.len()`, so the
        // `WORD_BYTES` bytes read here are in bounds. A `[u8; WORD_BYTES]` read has no alignment
        // requirement, so the unaligned pointer cast is sound.
        let chunk =
            usize::from_le_bytes(unsafe { *(bytes.as_ptr().add(pos) as *const [u8; WORD_BYTES]) });
        // Computed once and reused: word-continue is alphanumeric plus `_`, `word_like` tracks the
        // alphanumeric lanes, and `upper` the `[A-Z]` lanes.
        let is_alpha = swar_in_range(chunk | (SWAR_ONES * 0x20), b'a', b'z');
        let alnum = is_alpha | swar_in_range(chunk, b'0', b'9');
        // Uppercase = alpha lanes whose `0x20` case bit is clear. `is_alpha` already lives in the
        // high bit of each lane; shifting the chunk left by 2 moves each lane's bit-5 (the `0x20`
        // case bit) into that same high bit (5 + 2 = 7, stays within the lane), and `& !…` keeps
        // the lanes where it was clear. Derived from `is_alpha` with no extra range test, so the
        // uppercase signal rides along the load this scan already does.
        let upper = is_alpha & !(chunk << 2) & SWAR_HIGH;
        let cont = alnum | swar_in_range(chunk, b'_', b'_');
        let boundary = (!cont) & SWAR_HIGH;
        if boundary == 0 {
            // All `WORD_BYTES` bytes continue the word.
            word_like |= alnum != 0;
            has_upper |= upper != 0;
            pos += WORD_BYTES;
        } else {
            // First non-continue byte is at this offset within the chunk.
            let off = (boundary.trailing_zeros() / 8) as usize;
            // Only the lanes *before* the boundary are part of the run.
            let consumed_mask = boundary & boundary.wrapping_neg(); // lowest boundary high-bit
            word_like |= (alnum & (consumed_mask - 1)) != 0;
            has_upper |= (upper & (consumed_mask - 1)) != 0;
            return (pos + off, word_like, has_upper);
        }
    }

    // Scalar tail (and small inputs): same classification, one byte at a time.
    while pos < bytes.len() {
        let b = bytes[pos];
        let is_alnum = is_ascii_alnum(b);
        if !(is_alnum | (b == b'_')) {
            break;
        }
        word_like |= is_alnum;
        has_upper |= b.is_ascii_uppercase();
        pos += 1;
    }
    (pos, word_like, has_upper)
}

/// Number of bytes classified per SWAR iteration (the platform word size).
const WORD_BYTES: usize = size_of::<usize>();
/// High bit of every byte lane.
const SWAR_HIGH: usize = usize::from_ne_bytes([0x80; WORD_BYTES]);
/// `0x01` in every byte lane.
const SWAR_ONES: usize = usize::from_ne_bytes([0x01; WORD_BYTES]);

/// Branchless `[a-zA-Z0-9]` test for a single ASCII byte (false for bytes >= 0x80).
#[inline(always)]
fn is_ascii_alnum(b: u8) -> bool {
    let is_alpha = (b | 0x20).wrapping_sub(b'a') < 26;
    let is_digit = b.wrapping_sub(b'0') < 10;
    is_alpha | is_digit
}

/// Per-byte `lo <= byte <= hi` (requires `0 <= lo <= hi <= 0x7F`): sets each lane's high bit when
/// in range. Bytes >= 0x80 never match.
///
/// Carry-safe: the comparisons run on the low 7 bits of each lane, so every per-lane addition
/// stays <= 0xFF and never carries into the neighboring lane. Lanes whose byte is >= 0x80 are
/// masked out via `& !x` at the end.
#[inline(always)]
fn swar_in_range(x: usize, lo: u8, hi: u8) -> usize {
    // Outside this range the `0x80 - lo` / `0x7F - hi` broadcasts below would over/underflow and
    // the per-lane adds could carry across lanes, silently misclassifying bytes.
    debug_assert!(lo <= hi && hi <= 0x7F);
    let lo7 = x & !SWAR_HIGH;
    // High bit set iff lo7 >= lo  (lo7 + (0x80 - lo) reaches 0x80 exactly when lo7 >= lo).
    let ge_lo = lo7.wrapping_add(SWAR_ONES * (0x80 - lo as usize));
    // High bit set iff lo7 >  hi  (lo7 + (0x7F - hi) reaches 0x80 exactly when lo7 > hi).
    let gt_hi = lo7.wrapping_add(SWAR_ONES * (0x7F - hi as usize));
    ge_lo & !gt_hi & !x & SWAR_HIGH
}

/// Folds a freshly-scanned ASCII word run into the in-progress token: ORs its word-like / uppercase
/// contribution into `token_props`, and returns the DFA state the run's last byte lands in
/// (`b'0'..=b'9'` → Numeric, `b'_'` → ExtendNumLet, else ALetter).
///
/// The single source of truth for both halves of that mapping. The fast lane and the per-char ASCII
/// branch both route through it, so they cannot disagree on a byte's contribution, and
/// `entry_fast_path_assumptions_hold_in_table` checks the state half against `TABLE` directly.
/// `#[inline(always)]`: it's in the tokenizer's hottest loop, and an out-of-line call here would
/// break the surrounding inlining (the crate has hit that cliff before).
#[inline(always)]
fn fold_ascii_word(
    token_props: &mut TokenProperties,
    word_like: bool,
    has_upper: bool,
    last_byte: u8,
) -> State {
    if word_like {
        *token_props |= TokenProperties::WORD_LIKE;
    }
    if has_upper {
        *token_props |= TokenProperties::HAS_ASCII_UPPER;
    }
    match last_byte {
        b'0'..=b'9' => State::Numeric,
        b'_' => State::ExtendNumLet,
        _ => State::ALetter,
    }
}

/// Cheap-path `TokenProperties` contribution for each `WordBreakProperty` value. Covers the
/// signals that fall out of WordBreak alone — letters and digits. Katakana is intentionally
/// **not** included: its set mixes Katakana letters (word-like) with the prolonged-sound mark
/// `ー` (Script=Common, not word-like). Those split is resolved via `is_word_like_strict`.
const WORD_BREAK_CONTRIB: [TokenProperties; WordBreakProperty::NUM_VARIANTS] = {
    let mut t = [TokenProperties(0); WordBreakProperty::NUM_VARIANTS];
    t[WordBreakProperty::ALetter as usize] = TokenProperties::WORD_LIKE;
    t[WordBreakProperty::HebrewLetter as usize] = TokenProperties::WORD_LIKE;
    t[WordBreakProperty::Numeric as usize] = TokenProperties::WORD_LIKE;
    t
};

#[cfg(test)]
mod tests {
    use super::{Options, scan_word_continue, tokenize};
    use crate::uax29::test_helpers::test_against_uax29_break_tests;

    /// The ASCII fast path replaces a run of per-char DFA steps with one scan, so it hardcodes two
    /// facts about `TABLE`/`ASCII_WORD_BREAK_PROP` (the single source of truth): from every state the
    /// fast path runs in, each word-continue byte is a `NoBreak`, and the state it lands in is the
    /// one `fold_ascii_word` derives from the run's last byte. Assert both against the table
    /// directly, so a future table edit can't silently desync the fast path from the DFA it
    /// shortcuts — the scan skips those rows entirely and no output test would localize the break.
    #[test]
    fn entry_fast_path_assumptions_hold_in_table() {
        use super::properties::ASCII_WORD_BREAK_PROP;
        use super::transitions::{State, TABLE, Transition};
        use super::{TokenProperties, fold_ascii_word, is_ascii_alnum};
        use crate::uax29::Action;

        for state in [
            State::ALetter,
            State::Numeric,
            State::ExtendNumLet,
            State::HLetter,
        ] {
            for b in 0u8..0x80 {
                // Exactly `scan_word_continue`'s continuation predicate; other bytes stop the scan
                // and the DFA handles them, so the table may do anything there.
                if !(is_ascii_alnum(b) || b == b'_') {
                    continue;
                }
                let Transition(next_state, action) =
                    TABLE[state as usize][ASCII_WORD_BREAK_PROP[b as usize] as usize];
                assert!(
                    matches!(action, Action::NoBreak),
                    "{state:?} x {:?} must NoBreak for the scan to consume it",
                    b as char
                );
                let mut p = TokenProperties::default();
                let folded = fold_ascii_word(&mut p, is_ascii_alnum(b), b.is_ascii_uppercase(), b);
                assert_eq!(
                    next_state, folded,
                    "{state:?} x {:?}: DFA lands in {next_state:?}, fold_ascii_word says {folded:?}",
                    b as char
                );
            }
        }
    }

    /// `Action::Break` is the one arm that does *not* clear `deferred_break_pos`/`deferred_props`,
    /// which is only sound because a break can never be reached from a deferred state. That gives
    /// the invariant `deferred_break_pos.is_some() ⟺ state.is_deferred()`, and the ASCII fast path
    /// depends on it: the fast path skips the `NoBreak` arm's deferred bookkeeping entirely, so were
    /// a break to leave a deferred position behind, the fast path would carry it past the point the
    /// DFA would have cleared it and misattribute a later token's properties.
    ///
    /// Nothing local to `tokenize` expresses this, so assert it on the table.
    #[test]
    fn deferred_states_never_break() {
        use super::properties::WordBreakProperty;
        use super::transitions::{State, TABLE, Transition};
        use crate::uax29::Action;

        for &state in State::ALL.iter() {
            if !state.is_deferred() {
                continue;
            }
            for prop in 0..WordBreakProperty::NUM_VARIANTS {
                let Transition(_, action) = TABLE[state as usize][prop];
                assert!(
                    !matches!(action, Action::Break),
                    "deferred {state:?} x prop {prop} is a Break, which would strand \
                     deferred_break_pos and break the ASCII fast path's invariant"
                );
            }
        }
    }

    /// Trivially-correct reference: byte-at-a-time `[a-zA-Z0-9_]` scan.
    fn scan_reference(bytes: &[u8], start: usize) -> (usize, bool, bool) {
        let mut pos = start;
        let mut word_like = false;
        let mut has_upper = false;
        while pos < bytes.len() {
            let b = bytes[pos];
            let is_alnum = b.is_ascii_alphanumeric();
            if !(is_alnum || b == b'_') {
                break;
            }
            word_like |= is_alnum;
            has_upper |= b.is_ascii_uppercase();
            pos += 1;
        }
        (pos, word_like, has_upper)
    }

    #[test]
    fn scan_word_continue_matches_reference() {
        // Exhaustive: every single byte value, at every alignment offset within a chunk, with a
        // word-continue prefix so the SWAR lane that contains the byte varies.
        for prefix in 0..=16usize {
            for b in 0..=255u8 {
                let mut buf = vec![b'a'; prefix];
                buf.push(b);
                buf.extend_from_slice(b"z9_more");
                assert_eq!(
                    scan_word_continue(&buf, 0),
                    scan_reference(&buf, 0),
                    "prefix={prefix} byte={b:#04x}"
                );
            }
        }

        // Randomized multi-byte inputs across the full byte range (biased toward word chars so we
        // exercise long runs and boundaries at every offset).
        let mut state: u64 = 0x9E3779B97F4A7C15;
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
                        (r >> 8) as u8 // full range, includes non-ASCII & punctuation
                    } else {
                        let alphabet = b"abcXYZ0189_ .,";
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
                scan_word_continue(&buf, start),
                scan_reference(&buf, start),
                "buf={buf:?} start={start}"
            );
        }
    }

    #[test]
    fn test_word_break_against_uax29_tests() {
        let (passed, failed) =
            test_against_uax29_break_tests("testdata/WordBreakTest.txt", |s, breakpoints| {
                tokenize(s, Options::default(), |bp, _props| {
                    breakpoints.push(bp);
                    true
                });
            });
        assert_eq!(
            (1944, 0),
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
            tokenize(s, Options::default(), |bp, _props| {
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
        assert_breaks("\n", vec![0, 1]);

        // WB5: ALetter × ALetter
        assert_breaks("hello", vec![0, 5]);

        // WB8: Numeric × Numeric
        assert_breaks("123", vec![0, 3]);

        // WB9/WB10: ALetter × Numeric, Numeric × ALetter
        assert_breaks("abc123", vec![0, 6]);
        assert_breaks("123abc", vec![0, 6]);
        assert_breaks("a1b2", vec![0, 4]);

        // WB3: CR × LF (stay together)
        assert_breaks("\r\n", vec![0, 2]);
        assert_breaks("\r\n\r\n", vec![0, 2, 4]);

        // CR and LF alone break normally
        assert_breaks("\r", vec![0, 1]);
        assert_breaks("\n\n", vec![0, 1, 2]);

        // Mixed with newlines
        assert_breaks("a\r\nb", vec![0, 1, 3, 4]);
        assert_breaks("ab\r\ncd", vec![0, 2, 4, 6]);

        // Keep horizontal whitespace together (WB3d)
        assert_breaks("a   c", vec![0, 1, 4, 5]);

        // Do not break letters across certain punctuation, such as within "e.g." or "example.com".
        assert_breaks("e.g. hello", vec![0, 3, 4, 5, 10]);
        assert_breaks("example.com", vec![0, 11]);
        assert_breaks("won't", vec![0, 5]);

        // WB13a/WB13b: ExtendNumLet connects letters, numbers, katakana
        assert_breaks("a_1", vec![0, 3]);
        assert_breaks("_a", vec![0, 2]);

        // Edge cases with deferred breaks.
        assert_breaks("can'", vec![0, 3, 4]);
        assert_breaks("can' hi", vec![0, 3, 4, 5, 7]);

        // WB7a and WB6/WB7 with Hebrew_Letter and Single_Quote.
        assert_breaks("א'", vec![0, "א'".len()]);
        assert_breaks("א'א", vec![0, "א'א".len()]);
        assert_breaks("א'\u{2060}א", vec![0, "א'\u{2060}א".len()]);
        assert_breaks("א'a", vec![0, "א'a".len()]);
        assert_breaks("הצ'קרות", vec![0, "הצ'קרות".len()]);
        assert_breaks(
            "לייף אנרג'י",
            vec![0, "לייף".len(), "לייף ".len(), "לייף אנרג'י".len()],
        );

        // WB7b/WB7c: Hebrew_Letter × Double_Quote × Hebrew_Letter (gershayim acronyms
        // like צה״ל). With letters on both sides the gershayim is absorbed into the
        // word; with whitespace on either side it must emit as its own standalone
        // token (UAX #29 prescribes a break — no MidLetter/DoubleQuote rule applies).
        assert_breaks("צה\u{05F4}ל", vec![0, "צה\u{05F4}ל".len()]);
        // Closing gershayim followed by space: standalone token.
        assert_breaks(
            "אקספרס\u{05F4} מהיום",
            vec![
                0,
                "אקספרס".len(),
                "אקספרס\u{05F4}".len(),
                "אקספרס\u{05F4} ".len(),
                "אקספרס\u{05F4} מהיום".len(),
            ],
        );
        // Full quoted-word pattern: both opening and closing gershayim are standalone.
        assert_breaks(
            "\u{05F4}אקספרס\u{05F4} מהיום",
            vec![
                0,
                "\u{05F4}".len(),
                "\u{05F4}אקספרס".len(),
                "\u{05F4}אקספרס\u{05F4}".len(),
                "\u{05F4}אקספרס\u{05F4} ".len(),
                "\u{05F4}אקספרס\u{05F4} מהיום".len(),
            ],
        );

        // WB3c: ZWJ × Extended_Pictographic (emoji ZWJ sequences)
        assert_breaks("👨\u{200D}👩", vec![0, 11]);
        assert_breaks("👨👩", vec![0, 4, 8]);

        // Weird edge case: Letters that are also extended pictographic
        assert_breaks("🇦", vec![0, 4]);
        assert_breaks("🇦🇦", vec![0, 8]);
        assert_breaks("🇦🇦🇦", vec![0, 8, 12]);

        // Circled letters
        assert_breaks("\u{200d}Ⓜ", vec![0, 6]);
    }

    #[test]
    fn tokenizer_properties_sanity() {
        // Each emit reports properties of the span just closed; the leading boundary at 0 has
        // no preceding span, so it carries default props.
        fn assert_props(s: &str, expected: Vec<(usize, bool)>) {
            let mut got: Vec<(usize, bool)> = Vec::new();
            tokenize(s, Options::default(), |bp, props| {
                got.push((bp, props.is_ascii()));
                true
            });
            assert_eq!(got, expected, "input: {:?}", s);
        }

        // Leading boundary at 0 is vacuously is_ascii=true.
        assert_props("hello", vec![(0, true), (5, true)]);
        assert_props("🛑", vec![(0, true), (4, false)]);

        // The sharp case: the breaking char is non-ASCII but starts the *next* token, so "ab"
        // must still report is_ascii=true and "🛑" must report is_ascii=false.
        assert_props("ab🛑", vec![(0, true), (2, true), (6, false)]);
    }

    #[test]
    fn tokenizer_has_ascii_upper_sanity() {
        // Each emit reports properties of the span just closed; the leading boundary at 0 has
        // no preceding span, so has_ascii_upper is vacuously false.
        fn assert_has_ascii_upper(s: &str, expected: Vec<(usize, bool)>) {
            let mut got: Vec<(usize, bool)> = Vec::new();
            tokenize(s, Options::default(), |bp, props| {
                got.push((bp, props.has_ascii_upper()));
                true
            });
            assert_eq!(got, expected, "input: {:?}", s);
        }

        assert_has_ascii_upper("hello", vec![(0, false), (5, false)]);
        assert_has_ascii_upper("Hello", vec![(0, false), (5, true)]);
        assert_has_ascii_upper("HELLO", vec![(0, false), (5, true)]);
        assert_has_ascii_upper("aB", vec![(0, false), (2, true)]);
        assert_has_ascii_upper("123", vec![(0, false), (3, false)]);

        // The breaking char is non-ASCII but starts the *next* token, so "ab" must still
        // report has_ascii_upper=false.
        assert_has_ascii_upper("ab🛑", vec![(0, false), (2, false), (6, false)]);
    }

    fn assert_word_like(s: &str, expected: Vec<(usize, bool)>) {
        let mut got: Vec<(usize, bool)> = Vec::new();
        tokenize(s, Options::default(), |bp, props| {
            got.push((bp, props.is_word_like()));
            true
        });
        assert_eq!(got, expected, "input: {:?}", s);
    }

    /// ASCII subset of the word-like contract: any token containing an ASCII letter or digit is
    /// word-like; pure-connector / whitespace / punctuation tokens are not. The leading boundary
    /// at 0 has no preceding span, so word_like is vacuously false.
    #[test]
    fn tokenizer_word_like_ascii_sanity() {
        // ASCII letters / digits / mixed / contractions.
        assert_word_like("hello", vec![(0, false), (5, true)]);
        assert_word_like("123", vec![(0, false), (3, true)]);
        assert_word_like("abc123", vec![(0, false), (6, true)]);
        assert_word_like("won't", vec![(0, false), (5, true)]);

        // Connectors only (ExtendNumLet) — `_` is not a letter or digit.
        assert_word_like("___", vec![(0, false), (3, false)]);
        // Whitespace only.
        assert_word_like("   ", vec![(0, false), (3, false)]);
        // ASCII punctuation: each '!' breaks separately, none word-like.
        assert_word_like("!!!", vec![(0, false), (1, false), (2, false), (3, false)]);
    }

    /// Strict cases that need Script / Ideographic / OtherNumber / ExtPict lookups beyond the
    /// WordBreak property.
    #[test]
    fn tokenizer_word_like_strict_sanity() {
        // Hebrew (HebrewLetter prop)
        assert_word_like("ש", vec![(0, false), (2, true)]);

        // CJK ideograph: WordBreak=Other, Script=Han.
        assert_word_like("中", vec![(0, false), (3, true)]);
        // Ideographic iteration mark: WordBreak=Other, Script=Common, Ideographic=true.
        assert_word_like("々", vec![(0, false), (3, true)]);
        // Circled digit: WordBreak=Other, GeneralCategory=OtherNumber.
        assert_word_like("①", vec![(0, false), (3, true)]);
        // Devanagari letter: WordBreak=Other, Script=Devanagari.
        assert_word_like("अ", vec![(0, false), (3, true)]);
        // Thai letter: WordBreak=Other, Script=Thai.
        assert_word_like("ก", vec![(0, false), (3, true)]);
        // Emoji: WordBreak=Other (or ExtPict), Script=Common, ExtendedPictographic=true.
        assert_word_like("👍", vec![(0, false), (4, true)]);

        // Real Katakana letter: WordBreak=Katakana, Script=Katakana → word-like.
        assert_word_like("リ", vec![(0, false), (3, true)]);
        // Katakana-Hiragana extender: WordBreak=Katakana, Script=Common → NOT word-like.
        // Locks in why we can't just OR `WordBreakProperty::Katakana → WORD_LIKE`; we need to
        // additionally check the char's Script.
        assert_word_like("ー", vec![(0, false), (3, false)]);

        // HEBREW PUNCTUATION GERSHAYIM (U+05F4): WordBreak=DoubleQuote (not word-like via
        // cheap path), Script=Hebrew (word-like via strict). Standalone token is word-like.
        assert_word_like("\u{05F4}", vec![(0, false), ("\u{05F4}".len(), true)]);
    }

    /// A deferred break must not strand the deferred char's properties on the preceding
    /// token. For `אקספרס״ `, the closing gershayim emerges as a standalone token via
    /// `DeferredBreak` from `HLetterDQ`; its `WORD_LIKE` bit (via Script=Hebrew) belongs
    /// to that standalone token, not to the Hebrew word that precedes it.
    #[test]
    fn deferred_break_does_not_misattribute_props() {
        let s = "אקספרס\u{05F4} ";
        assert_word_like(
            s,
            vec![
                (0, false),
                ("אקספרס".len(), true),         // אקספרס (Hebrew letters)
                ("אקספרס\u{05F4}".len(), true), // ״ standalone — Script=Hebrew
                (s.len(), false),               // trailing space
            ],
        );

        // Same shape with Hebrew word after the space — the four-quote pattern from
        // real Common Crawl docs (`״אקספרס״ מהיום …`). All four standalone gershayim
        // tokens must be word-like; the asymmetry-bug case is the trailing one.
        let s = "\u{05F4}אקספרס\u{05F4} מהיום";
        assert_word_like(
            s,
            vec![
                (0, false),
                ("\u{05F4}".len(), true),                 // leading ״
                ("\u{05F4}אקספרס".len(), true),           // אקספרס
                ("\u{05F4}אקספרס\u{05F4}".len(), true),   // trailing ״
                ("\u{05F4}אקספרס\u{05F4} ".len(), false), // space
                (s.len(), true),                          // מהיום
            ],
        );
    }
}
