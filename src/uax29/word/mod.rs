pub(crate) mod properties;
pub(crate) mod transitions;

use crate::uax29::Action;
use crate::uax29::swar::{
    SWAR_HIGH, SWAR_ONES, WORD_BYTES, first_boundary_byte, load_chunk, swar_in_range,
};
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
        // Fast path for ASCII: consume a whole word-continue run at once and skip the DFA entirely.
        //
        // `is_ascii_word_start(bytes[pos])` is exactly `scan_word_continue`'s own continuation
        // predicate (alnum-or-underscore, identical across all three backends), so it tells us
        // upfront whether the scan below would make any progress at all — without it, `state`
        // matching here but the current byte not continuing (a space, a non-ASCII lead byte, a
        // ZWJ...) would still call `scan_word_continue` only to load and classify a chunk and
        // discover zero progress, every time that happens.
        if matches!(
            state,
            State::ALetter | State::Numeric | State::ExtendNumLet | State::HLetter
        ) && is_ascii_word_start(bytes[pos])
        {
            let scan_start = pos;
            let (end, word_like, has_upper) = scan_word_continue(bytes, pos);
            pos = end;
            if pos > scan_start {
                // `bytes[pos - 1]` is in bounds: `pos > scan_start`, so `pos >= 1`.
                state = fold_ascii_word(&mut token_props, word_like, has_upper, bytes[pos - 1]);
                last_was_zwj = false;

                // Fused fast-path for the dominant `word · spaces · word` alternation. A word
                // followed by a run of plain spaces (U+0020) is an *unconditional* break pair under
                // UAX#29 — `word ÷ WSegSpace` and `WSegSpace ÷ next` — with the whole run a single
                // WSegSpace segment (WB3d). Emitting both breaks inline and rolling straight into the
                // next word keeps that alternation out of the per-char DFA, whose cell-load +
                // indirect jump is the bulk of non-scan word-break time (profiled). Only U+0020
                // qualifies (tab/newline/NBSP break differently); and only when a fast-scannable word
                // follows the spaces — otherwise leave `pos`/`state` untouched and let the DFA emit
                // the WSegSpace break, since its post-break state depends on the following char.
                while pos < text.len() && bytes[pos] == b' ' {
                    let word_end = pos;
                    let mut after = pos;
                    while after < text.len() && bytes[after] == b' ' {
                        after += 1;
                    }
                    if after >= text.len() || !is_ascii_word_start(bytes[after]) {
                        break; // DFA handles the space run and its terminator
                    }
                    // Word token ends at the first space.
                    if !on_breakpoint(word_end, std::mem::take(&mut token_props)) {
                        return;
                    }
                    // Space-run token ends at the next word start; its props are empty (ASCII, not
                    // word-like), so the take leaves `token_props` cleared for the next word.
                    pos = after;
                    if !on_breakpoint(pos, std::mem::take(&mut token_props)) {
                        return;
                    }
                    // Scan the next word inline (non-empty: `bytes[pos]` is a word-start byte).
                    let (e, wl, hu) = scan_word_continue(bytes, pos);
                    pos = e;
                    state = fold_ascii_word(&mut token_props, wl, hu, bytes[pos - 1]);
                }
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
/// [`TokenProperties::has_ascii_upper`]. The uppercase flag is computed in the same pass that's
/// already touching these bytes, so a case-folding consumer never needs a second per-token scan.
///
/// This is the tokenizer's hottest loop on Latin-script text. The backend is chosen entirely at
/// compile time via `cfg`, never at runtime: every backend here is part of its target's baseline ABI,
/// and a runtime `target_feature` boundary can't inline into this loop — it measured a net regression
/// even where the wider kernel was faster.
///
/// - `aarch64`       → [`scan_word_continue_neon`] (NEON is baseline there)
/// - `x86_64`        → [`scan_word_continue_sse2`] (SSE2 is the x86-64 baseline)
/// - everything else → [`scan_word_continue_swar`] (portable: wasm32, 32-bit, exotic targets)
///
/// Every backend is differentially tested against the same byte-at-a-time scalar reference, so they
/// are bit-identical by construction.
///
/// The two vector backends are deliberately *not* one shape, because the tradeoff turns on the cost
/// of a movemask (vector→GPR transfer). On NEON that is the multi-cycle `shrn`+`umov` trick, so
/// [`scan_word_continue_neon`] spends a cheap SWAR prefix to keep short words off the vector path and
/// defers its lane reductions to amortize the transfer. On x86 a `pmovmskb` is a single cheap uop, so
/// [`scan_word_continue_sse2`] stays naive — both tricks were measured to *regress* it. Sharing is
/// limited to what is genuinely identical: the SWAR classification kernel ([`swar_classify_word`],
/// reused by the NEON prefix) and the scalar tail ([`scan_word_continue_tail`], reused by all three).
#[inline]
fn scan_word_continue(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is guaranteed on every `aarch64` target (it is part of the base ISA), so the
    // intrinsics are always available without runtime detection.
    return unsafe { scan_word_continue_neon(bytes, start) };
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is guaranteed on every `x86_64` target (it is part of the x86-64 baseline),
    // so the intrinsics are always available without runtime detection.
    return unsafe { scan_word_continue_sse2(bytes, start) };
    // SWAR fallback: non-SIMD targets (wasm32, 32-bit, …).
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    return scan_word_continue_swar(bytes, start);
}

/// Classifies one SWAR word (`WORD_BYTES` bytes, already loaded as a little-endian `usize`) for the
/// word-continue scan. Returns the per-lane `alnum` mask (alphanumeric lanes), the `upper` mask
/// (`[A-Z]` lanes), and the `boundary` mask (the lane high bit is set in every non-continue lane).
/// Shared by [`scan_word_continue_swar`] and the SWAR prefix of [`scan_word_continue_neon`] so the
/// two cannot drift — a silent classification divergence would be a correctness bug, caught only by
/// tests otherwise.
// Only the SWAR core (dead on `aarch64`/`x86_64` outside tests/bench) and the aarch64 NEON prefix
// call this, so mirror the core's dead-code allowance on the SIMD targets.
#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn swar_classify_word(chunk: usize) -> (usize, usize, usize) {
    // word-continue is alphanumeric plus `_`; `alnum` tracks the alphanumeric lanes, `upper` the
    // `[A-Z]` lanes.
    let is_alpha = swar_in_range(chunk | (SWAR_ONES * 0x20), b'a', b'z');
    let alnum = is_alpha | swar_in_range(chunk, b'0', b'9');
    // Uppercase = alpha lanes whose `0x20` case bit is clear. `is_alpha` already lives in the high
    // bit of each lane; shifting the chunk left by 2 moves each lane's bit-5 (the `0x20` case bit)
    // into that same high bit (5 + 2 = 7, stays within the lane), and `& !…` keeps the lanes where
    // it was clear. Derived from `is_alpha` with no extra range test, so the uppercase signal rides
    // along the load this scan already does.
    let upper = is_alpha & !(chunk << 2) & SWAR_HIGH;
    let cont = alnum | swar_in_range(chunk, b'_', b'_');
    let boundary = (!cont) & SWAR_HIGH;
    (alnum, upper, boundary)
}

/// Resolves a SWAR word-continue boundary hit (`boundary != 0`): folds `alnum`/`upper` into the
/// accumulators, masked to the lanes before the boundary, and returns the final scan result. Shared
/// by [`scan_word_continue_swar`] and the SWAR prefix in [`scan_word_continue_neon`] — the only two
/// places that resolve this exact `(alnum, upper, boundary)` shape.
#[inline(always)]
fn resolve_swar_word_boundary(
    pos: usize,
    boundary: usize,
    alnum: usize,
    upper: usize,
    mut word_like: bool,
    mut has_upper: bool,
) -> (usize, bool, bool) {
    let off = first_boundary_byte(boundary);
    let consumed_mask = boundary & boundary.wrapping_neg(); // lowest boundary high-bit
    word_like |= (alnum & (consumed_mask - 1)) != 0;
    has_upper |= (upper & (consumed_mask - 1)) != 0;
    (pos + off, word_like, has_upper)
}

/// Portable SWAR (SIMD-within-a-register) core: classifies a `usize` word (8 bytes on 64-bit, 4 on
/// wasm32) per iteration with pure integer arithmetic — no per-byte table load (removes a load→load
/// dependency that capped the original loop at ~1 cycle/byte) and no architecture intrinsics. This
/// is the universal fallback and stays exercised on every host via its own differential test.
// On `aarch64`/`x86_64` the dispatcher never calls this (SIMD wins), so outside tests it is only
// the fallback arm for other targets — silence dead-code there.
#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn scan_word_continue_swar(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    let mut pos = start;
    let mut word_like = false;
    let mut has_upper = false;

    // SWAR fast lane: classify `WORD_BYTES` bytes per iteration. We only enter the word-at-a-time
    // path while a full word remains; the scalar tail below finishes the run.
    while pos + WORD_BYTES <= bytes.len() {
        // SAFETY: the `while` condition guarantees `pos + WORD_BYTES <= bytes.len()`.
        let chunk = unsafe { load_chunk(bytes, pos) };
        let (alnum, upper, boundary) = swar_classify_word(chunk);
        if boundary == 0 {
            // All `WORD_BYTES` bytes continue the word.
            word_like |= alnum != 0;
            has_upper |= upper != 0;
            pos += WORD_BYTES;
        } else {
            return resolve_swar_word_boundary(pos, boundary, alnum, upper, word_like, has_upper);
        }
    }

    // Scalar tail (and small inputs): finish one byte at a time — see `scan_word_continue_tail`.
    scan_word_continue_tail(bytes, pos, word_like, has_upper)
}

/// Lane count of the vector backends. NEON (`uint8x16_t`) and SSE2/PSHUFB (`__m128i`) each process
/// a 16-byte register per iteration.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const SIMD_LANES: usize = 16;

/// Bytes a word-continue run must survive (boundary-free) in the cheap SWAR prefix before
/// [`scan_word_continue_neon`] escalates to the wide vector loop. Any word up to this length resolves
/// with no vector→GPR transfer, so all natural-language text stays on the SWAR-equivalent path and
/// only long runs (URLs, identifiers, base64) reach the wide loop.
///
/// It only takes effect at `WORD_BYTES` (8-byte) granularity, so this is coarser than it looks:
/// `16` means "two SWAR chunks" and every value in `9..=16` behaves identically. Worth re-sweeping
/// if the deployment target's movemask cost differs much from the Apple Silicon it was measured on;
/// setting it to `usize::MAX` makes the wide loop unreachable and reduces this backend to the
/// portable SWAR scan, which is the safe fallback if a future core regresses on it.
#[cfg(target_arch = "aarch64")]
const NEON_SWAR_PREFIX: usize = 16;

/// NEON backend. Classification mirrors [`scan_word_continue_swar`] lane-for-lane. Two structural
/// differences exist purely to dodge NEON's expensive vector→GPR transfers (a movemask on NEON is
/// the `shrn #4` trick plus a `umov`, multi-cycle and port-limited; that cost made a naive 16-byte
/// loop lose to SWAR on short, natural-language words):
///
///  1. SWAR prefix: the scan is entered once per word, and most words are short, so their boundary
///     lands inside the first 8-byte SWAR chunk, which resolves it with zero vector transfers. Only
///     a run that stays boundary-free past [`NEON_SWAR_PREFIX`] bytes (a long URL / identifier /
///     base64 blob, exactly where a 16-byte lane pays off) escalates to the wide loop.
///  2. Deferred reductions: in the wide loop, `alnum`/`upper` are accumulated as vectors and reduced
///     to scalars once on exit, so a steady-state chunk pays a single transfer (the boundary
///     movemask, needed for the break position) instead of three.
///
/// # Safety
/// Must be called on an `aarch64` target (NEON is baseline there, so no feature detection is
/// needed). `bytes`/`start` carry no extra invariant; the loads are bounds-checked internally.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)] // whole body is intrinsics; the fn's `# Safety` is the boundary
unsafe fn scan_word_continue_neon(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    use core::arch::aarch64::*;

    /// Packs a `0x00`/`0xFF`-per-lane mask into 4 bits per lane via the `shrn #4` trick, so lane
    /// `i` lands at bit `4*i`. `trailing_zeros() / 4` is then the first set lane.
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn movemask4(v: uint8x16_t) -> u64 {
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            v,
        ))))
    }

    const LANES: usize = SIMD_LANES;
    let mut pos = start;
    let mut word_like = false;
    let mut has_upper = false;

    // SWAR prefix (see this fn's doc). Uses the same per-chunk `swar_classify_word` as
    // `scan_word_continue_swar`; the only difference is that, instead of running to completion, it
    // hands off to the wide loop once a run survives `NEON_SWAR_PREFIX` bytes without a boundary.
    while pos + WORD_BYTES <= bytes.len() && pos - start < NEON_SWAR_PREFIX {
        // SAFETY: the `while` condition guarantees `pos + WORD_BYTES <= bytes.len()`.
        let chunk = load_chunk(bytes, pos);
        let (alnum, upper, boundary) = swar_classify_word(chunk);
        if boundary != 0 {
            return resolve_swar_word_boundary(pos, boundary, alnum, upper, word_like, has_upper);
        }
        word_like |= alnum != 0;
        has_upper |= upper != 0;
        pos += WORD_BYTES;
    }

    // Wide NEON loop with deferred reductions (see this fn's doc). Reached only for a long,
    // still-open run. `alnum`/`upper` fold into vector accumulators; the only per-chunk transfer
    // is the boundary movemask.
    let mut alnum_acc = vdupq_n_u8(0);
    let mut upper_acc = vdupq_n_u8(0);
    while pos + LANES <= bytes.len() {
        // SAFETY: the `while` condition guarantees `pos + LANES <= bytes.len()`; `vld1q_u8` is an
        // unaligned load.
        let chunk = vld1q_u8(bytes.as_ptr().add(pos));
        // is_alpha: lowercase the lane (`| 0x20`) then test `a..=z`. A byte >= 0x80 can't enter
        // `a..=z` after `| 0x20`, matching the SWAR core's `& !x` masking.
        let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
        let is_alpha = vandq_u8(
            vcgeq_u8(lowered, vdupq_n_u8(b'a')),
            vcleq_u8(lowered, vdupq_n_u8(b'z')),
        );
        let is_digit = vandq_u8(
            vcgeq_u8(chunk, vdupq_n_u8(b'0')),
            vcleq_u8(chunk, vdupq_n_u8(b'9')),
        );
        let alnum = vorrq_u8(is_alpha, is_digit);
        let cont = vorrq_u8(alnum, vceqq_u8(chunk, vdupq_n_u8(b'_')));
        let boundary = vmvnq_u8(cont);
        // upper = alpha lanes whose `0x20` case bit is clear (i.e. `[A-Z]`).
        let case_clear = vceqq_u8(vandq_u8(chunk, vdupq_n_u8(0x20)), vdupq_n_u8(0));
        let upper = vandq_u8(is_alpha, case_clear);

        let bound_bits = movemask4(boundary);
        if bound_bits == 0 {
            alnum_acc = vorrq_u8(alnum_acc, alnum);
            upper_acc = vorrq_u8(upper_acc, upper);
            pos += LANES;
        } else {
            let off = (bound_bits.trailing_zeros() / 4) as usize;
            // Keep only lanes *before* the boundary (`off` in `0..LANES`, so `off*4 < 64`). Fold the
            // deferred accumulators (a horizontal max → nonzero iff any prior full chunk set a lane)
            // together with this final partial chunk. The extra movemask(s) here are paid once per
            // long run, not per chunk, so the steady-state single-transfer property holds.
            let keep = (1u64 << (off * 4)) - 1;
            word_like |= vmaxvq_u8(alnum_acc) != 0 || (movemask4(alnum) & keep) != 0;
            has_upper |= vmaxvq_u8(upper_acc) != 0 || (movemask4(upper) & keep) != 0;
            return (pos + off, word_like, has_upper);
        }
    }

    // Reduce the deferred accumulators once before the scalar tail picks up the remaining bytes.
    word_like |= vmaxvq_u8(alnum_acc) != 0;
    has_upper |= vmaxvq_u8(upper_acc) != 0;

    // Scalar tail (and small inputs): finish one byte at a time — see `scan_word_continue_tail`.
    scan_word_continue_tail(bytes, pos, word_like, has_upper)
}

/// SSE2 backend (16 bytes/iter). Same lane-for-lane classification as [`scan_word_continue_swar`].
/// Unsigned byte range tests are built from saturating subtraction (SSE2 has only *signed* byte
/// compares), and the per-lane reductions use `_mm_movemask_epi8` (1 bit/lane). Kept deliberately
/// naive (no SWAR prefix, no deferred reductions) because `pmovmskb` is cheap on x86; see the
/// movemask-cost discussion on [`scan_word_continue`] and the inline note below.
///
/// # Safety
/// Must be called on an `x86_64` target (SSE2 is baseline there, so no feature detection is
/// needed). `bytes`/`start` carry no extra invariant; the loads are bounds-checked internally.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)] // whole body is intrinsics; the fn's `# Safety` is the boundary
unsafe fn scan_word_continue_sse2(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    use core::arch::x86_64::*;

    /// `lo <= b <= hi` (unsigned) per lane → `0xFF`/`0x00`, via saturating subtraction: a lane is
    /// out of range iff `(lo - b)` or `(b - hi)` saturates above zero.
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn in_range(b: __m128i, lo: u8, hi: u8) -> __m128i {
        let below = _mm_subs_epu8(_mm_set1_epi8(lo as i8), b);
        let above = _mm_subs_epu8(b, _mm_set1_epi8(hi as i8));
        _mm_cmpeq_epi8(_mm_or_si128(below, above), _mm_setzero_si128())
    }

    const LANES: usize = SIMD_LANES;
    let mut pos = start;
    let mut word_like = false;
    let mut has_upper = false;

    // No deferred reductions / SWAR prefix here (unlike NEON): x86's `pmovmskb` is a cheap single
    // uop, so a per-chunk movemask costs little and the naive form measured fastest end-to-end.
    // Deferring the alnum/upper reductions was tried and *regressed* short words (~9% at len 5-12,
    // ~1.5% end-to-end) while only helping >64-byte runs — the reduction's upside needs NEON's
    // expensive vector→GPR transfer to pay off, which x86 doesn't have. See `scan_word_continue_neon`.
    while pos + LANES <= bytes.len() {
        // SAFETY: the `while` condition guarantees `pos + LANES <= bytes.len()`; `loadu` is an
        // unaligned load.
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(pos) as *const __m128i);
        // is_alpha: lowercase the lane (`| 0x20`) then test `a..=z`. A byte >= 0x80 can't enter
        // `a..=z` after `| 0x20`, matching the SWAR core's `& !x` masking.
        let lowered = _mm_or_si128(chunk, _mm_set1_epi8(0x20));
        let is_alpha = in_range(lowered, b'a', b'z');
        let is_digit = in_range(chunk, b'0', b'9');
        let alnum = _mm_or_si128(is_alpha, is_digit);
        let cont = _mm_or_si128(alnum, _mm_cmpeq_epi8(chunk, _mm_set1_epi8(b'_' as i8)));
        // boundary = !cont
        let boundary = _mm_andnot_si128(cont, _mm_set1_epi8(-1));

        let bound_bits = _mm_movemask_epi8(boundary) as u32;
        let alnum_bits = _mm_movemask_epi8(alnum) as u32;
        // upper = alpha lanes whose `0x20` case bit is clear (i.e. `[A-Z]`).
        let case_clear = _mm_cmpeq_epi8(
            _mm_and_si128(chunk, _mm_set1_epi8(0x20)),
            _mm_setzero_si128(),
        );
        let upper_bits = _mm_movemask_epi8(_mm_and_si128(is_alpha, case_clear)) as u32;

        if bound_bits == 0 {
            word_like |= alnum_bits != 0;
            has_upper |= upper_bits != 0;
            pos += LANES;
        } else {
            let off = bound_bits.trailing_zeros() as usize;
            // Keep only lanes *before* the boundary (`off` in `0..LANES`).
            let keep = (1u32 << off) - 1;
            word_like |= (alnum_bits & keep) != 0;
            has_upper |= (upper_bits & keep) != 0;
            return (pos + off, word_like, has_upper);
        }
    }

    // Scalar tail (and small inputs): finish one byte at a time — see `scan_word_continue_tail`.
    scan_word_continue_tail(bytes, pos, word_like, has_upper)
}

/// Scalar tail shared by all three backends (SWAR / NEON / SSE2): finishes a word-continue
/// run one byte at a time from `pos`, folding into the `word_like` / `has_upper` accumulators.
/// Handles the bytes left over after the last full chunk and inputs shorter than one chunk.
/// `#[inline(always)]` so each backend keeps identical codegen to the inlined loop it replaced.
#[inline(always)]
fn scan_word_continue_tail(
    bytes: &[u8],
    mut pos: usize,
    mut word_like: bool,
    mut has_upper: bool,
) -> (usize, bool, bool) {
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

/// Branchless `[a-zA-Z0-9]` test for a single ASCII byte (false for bytes >= 0x80).
#[inline(always)]
fn is_ascii_alnum(b: u8) -> bool {
    let is_alpha = (b | 0x20).wrapping_sub(b'a') < 26;
    let is_digit = b.wrapping_sub(b'0') < 10;
    is_alpha | is_digit
}

/// First-byte test for the word-continue run `[a-zA-Z0-9_]`. Used by the fused word·space fast-path
/// to decide whether the byte after a space run starts a word it can scan inline.
#[inline(always)]
fn is_ascii_word_start(b: u8) -> bool {
    is_ascii_alnum(b) | (b == b'_')
}

/// Folds a freshly-scanned ASCII word run into the in-progress token: ORs its word-like / uppercase
/// contribution into `token_props`, and returns the DFA state the run's last byte lands in
/// (`b'0'..=b'9'` → Numeric, `b'_'` → ExtendNumLet, else ALetter).
///
/// The single source of truth for both halves of that mapping. All three ASCII paths route through
/// it — the scan-once entry, the fused word·space loop, and the per-char branch — so none of them can
/// disagree on a run's contribution, and `entry_fast_path_assumptions_hold_in_table` checks the state
/// half against `TABLE` directly. `#[inline(always)]`: it's in the tokenizer's hottest loop, and an
/// out-of-line call here would break the surrounding inlining (the crate has hit that cliff before).
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

    /// Runs the full differential battery (exhaustive single-byte-at-every-offset + 200k randomized
    /// inputs) against the trivial scalar reference. `scan` is the implementation under test; it
    /// must agree with `scan_reference` on every `(end, word_like, has_upper)` triple.
    fn assert_scan_matches_reference(scan: impl Fn(&[u8], usize) -> (usize, bool, bool)) {
        // Exhaustive: every single byte value, at every alignment offset within a chunk, with a
        // word-continue prefix so the SWAR/SIMD lane that contains the byte varies.
        for prefix in 0..=16usize {
            for b in 0..=255u8 {
                let mut buf = vec![b'a'; prefix];
                buf.push(b);
                buf.extend_from_slice(b"z9_more");
                assert_eq!(
                    scan(&buf, 0),
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
                scan(&buf, start),
                scan_reference(&buf, start),
                "buf={buf:?} start={start}"
            );
        }
    }

    #[test]
    fn scan_word_continue_matches_reference() {
        // The dispatcher (whichever backend the host selects) and the portable SWAR core must both
        // match the reference.
        assert_scan_matches_reference(scan_word_continue);
        assert_scan_matches_reference(super::scan_word_continue_swar);
    }

    /// The NEON backend must be bit-identical to the scalar reference (and therefore to SWAR).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn scan_word_continue_neon_matches_reference() {
        assert_scan_matches_reference(|b, s| unsafe { super::scan_word_continue_neon(b, s) });
    }

    /// The SSE2 backend must be bit-identical to the scalar reference (and therefore to SWAR).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn scan_word_continue_sse2_matches_reference() {
        assert_scan_matches_reference(|b, s| unsafe { super::scan_word_continue_sse2(b, s) });
    }

    /// Locks in the fused `word · spaces · word` fast-path against the same boundaries the
    /// per-char DFA produces. These cases all flow through the inline space handling (plain U+0020
    /// runs between fast-scannable words); the DFA must still own everything it bails on.
    #[test]
    fn fused_word_space_fast_path() {
        fn assert_breaks(s: &str, expected: Vec<usize>) {
            let mut breakpoints = Vec::new();
            tokenize(s, Options::default(), |bp, _props| {
                breakpoints.push(bp);
                true
            });
            assert_eq!(breakpoints, expected, "input: {s:?}");
        }

        // Single spaces between words: each word and each space run is its own segment.
        assert_breaks("a b c", vec![0, 1, 2, 3, 4, 5]);
        assert_breaks("ab cd ef", vec![0, 2, 3, 5, 6, 8]);
        // Multi-space run stays one segment (WB3d), then the fused path rolls into the next word.
        assert_breaks("ab   cd", vec![0, 2, 5, 7]);
        // Trailing space run is emitted by the WB2 tail, not double-emitted by the fused loop.
        assert_breaks("ab ", vec![0, 2, 3]);
        assert_breaks("ab   ", vec![0, 2, 5]);
        // Next word starts with a digit / underscore: state + word-likeness come from the scan.
        assert_breaks("ab 12", vec![0, 2, 3, 5]);
        assert_breaks("ab _c", vec![0, 2, 3, 5]);
        // Leading spaces are handled by the DFA (not the fused entry), and must still be correct.
        assert_breaks("  ab", vec![0, 2, 4]);
        // A non-space terminator must fall back to the DFA: tab is not WSegSpace, and the word may
        // join following punctuation (won't / e.g.) rather than break.
        assert_breaks("a\tb", vec![0, 1, 2, 3]);
        assert_breaks("ab c.d", vec![0, 2, 3, 6]);
    }

    /// The fused fast-path hardcodes three UAX#29 facts as inline logic. Assert them against the
    /// canonical `TABLE`/`ASCII_WORD_BREAK_PROP` (the single source of truth) so a future table
    /// edit can't silently desync the fast-path from the DFA it's meant to shortcut.
    #[test]
    fn fused_fast_path_assumptions_hold_in_table() {
        use super::is_ascii_alnum;
        use super::properties::ASCII_WORD_BREAK_PROP;
        use super::transitions::{State, TABLE, Transition};
        use crate::uax29::Action;

        // (1) From every word state the fused entry runs in, a space unconditionally breaks
        //     (`word ÷ WSegSpace`) and lands in WSegSpace — and the space byte carries no props
        //     (not alphanumeric, not uppercase).
        assert!(
            !is_ascii_alnum(b' ') && !b' '.is_ascii_uppercase(),
            "space carries no props"
        );
        for s in [
            State::ALetter,
            State::Numeric,
            State::ExtendNumLet,
            State::HLetter,
        ] {
            let Transition(next_state, action) =
                TABLE[s as usize][ASCII_WORD_BREAK_PROP[b' ' as usize] as usize];
            assert!(matches!(action, Action::Break), "{s:?} ÷ space must Break");
            assert_eq!(
                next_state,
                State::WSegSpace,
                "space lands in WSegSpace from {s:?}"
            );
        }
        // (2) A space run is a single segment: WSegSpace × WSegSpace stays, no break (WB3d).
        let Transition(next_state, action) =
            TABLE[State::WSegSpace as usize][ASCII_WORD_BREAK_PROP[b' ' as usize] as usize];
        assert!(
            matches!(action, Action::NoBreak),
            "space × space must not break (WB3d)"
        );
        assert_eq!(next_state, State::WSegSpace);
        // (3) The space run breaks before whatever word-start byte follows (`WSegSpace ÷ next`).
        for b in [b'a', b'Z', b'0', b'_'] {
            let Transition(_, action) =
                TABLE[State::WSegSpace as usize][ASCII_WORD_BREAK_PROP[b as usize] as usize];
            assert!(
                matches!(action, Action::Break),
                "WSegSpace ÷ {:?} must Break",
                b as char
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
