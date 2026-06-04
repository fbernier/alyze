pub mod sentence;
pub mod word;

/// Portable SWAR (SIMD-within-a-register) primitives shared by the word and sentence tokenizers.
///
/// These classify bytes a machine word at a time using pure integer arithmetic — no per-byte
/// table loads, no x86 intrinsics, no nightly `std::simd` — so they stay stable-Rust and
/// WASM-compatible while removing the load→load dependency that caps byte-at-a-time scanning.
pub(crate) mod swar {
    /// Number of bytes classified per SWAR iteration (the platform word size: 8 on 64-bit, 4 on
    /// wasm32).
    pub(crate) const WORD_BYTES: usize = size_of::<usize>();
    /// High bit of every byte lane.
    pub(crate) const SWAR_HIGH: usize = usize::from_ne_bytes([0x80; WORD_BYTES]);
    /// `0x01` in every byte lane.
    pub(crate) const SWAR_ONES: usize = usize::from_ne_bytes([0x01; WORD_BYTES]);

    /// Loads the `WORD_BYTES` bytes at `pos` as a little-endian `usize`. Reading little-endian
    /// regardless of host order keeps `trailing_zeros` pointing at the first (lowest-address) byte
    /// lane on big-endian targets too.
    ///
    /// # Safety
    /// `pos + WORD_BYTES <= bytes.len()` must hold.
    #[inline(always)]
    pub(crate) unsafe fn load_chunk(bytes: &[u8], pos: usize) -> usize {
        // A `[u8; WORD_BYTES]` read has no alignment requirement, so the unaligned cast is sound.
        usize::from_le_bytes(unsafe { *(bytes.as_ptr().add(pos) as *const [u8; WORD_BYTES]) })
    }

    /// Per-byte `lo <= byte <= hi` (requires `0 <= lo <= hi <= 0x7F`): sets each lane's high bit
    /// when in range. Bytes >= 0x80 never match.
    ///
    /// Carry-safe: the comparisons run on the low 7 bits of each lane, so every per-lane addition
    /// stays <= 0xFF and never carries into the neighboring lane. Lanes whose byte is >= 0x80 are
    /// masked out via `& !x` at the end.
    #[inline(always)]
    pub(crate) fn swar_in_range(x: usize, lo: u8, hi: u8) -> usize {
        // Outside this range the `0x80 - lo` / `0x7F - hi` broadcasts below would over/underflow
        // and the per-lane adds could carry across lanes, silently misclassifying bytes.
        debug_assert!(lo <= hi && hi <= 0x7F);
        let lo7 = x & !SWAR_HIGH;
        // High bit set iff lo7 >= lo  (lo7 + (0x80 - lo) reaches 0x80 exactly when lo7 >= lo).
        let ge_lo = lo7.wrapping_add(SWAR_ONES * (0x80 - lo as usize));
        // High bit set iff lo7 >  hi  (lo7 + (0x7F - hi) reaches 0x80 exactly when lo7 > hi).
        let gt_hi = lo7.wrapping_add(SWAR_ONES * (0x7F - hi as usize));
        ge_lo & !gt_hi & !x & SWAR_HIGH
    }

    /// Byte offset (within a SWAR chunk) of the first set high-bit lane in a boundary mask, e.g. as
    /// produced by [`swar_in_range`] or a classifier built on it. `mask` must be nonzero, with only
    /// each matched lane's high bit set (never other bits within a lane).
    #[inline(always)]
    pub(crate) fn first_boundary_byte(mask: usize) -> usize {
        (mask.trailing_zeros() / 8) as usize
    }
}

/// Helper to generate a state enum and associated constants.
/// Ensures `ALL` and `NUM_VARIANTS` are always in sync with the actual variants.
macro_rules! state_enum {
    ($($variant:ident),* $(,)?) => {
        #[repr(u8)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum State { $($variant),* }

        impl State {
            pub const ALL: &[Self] = &[$(Self::$variant),*];
            pub const NUM_VARIANTS: usize = Self::ALL.len();
        }
    };
}
pub(crate) use state_enum;

/// Helper to generate a break property enum and associated constants.
/// Ensures `ALL` and `NUM_VARIANTS` are always in sync with the actual variants.
macro_rules! break_property_enum {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[repr(u8)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),* }

        impl $name {
            pub(crate) const ALL: &[Self] = &[$(Self::$variant),*];
            pub(crate) const NUM_VARIANTS: usize = Self::ALL.len();
        }
    };
}
pub(crate) use break_property_enum;

/// Action is the result of a state transition. Each transition advances to a new state & emits
/// an action.
#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Action {
    Break,
    NoBreak,

    /// Defer break decisions. When this action is taken, we emit a break at the
    /// previously-stored deferred position and re-examine the current character in
    /// the new state (pos does not advance).
    ///
    /// Example (word): "can't" — when we hit the apostrophe, whether we break depends on the next
    /// character. If it's a letter, we don't break (WB6). If not, we break before the apostrophe.
    DeferredBreak,

    /// Make a character effectively invisible to the state machine: silently consume it and keep
    /// the same state. Used for WB4 (Extend/Format/ZWJ transparency) and SB5 (Format/Extend).
    /// When the action is `Transparent`, the state of the `Transition` is ignored.
    Transparent,
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use itertools::{EitherOrBoth, Itertools};

    /// A helper enum to represent the expected sequence of codepoints and break points in a test case.
    #[derive(Debug, PartialEq)]
    pub enum SequenceItem {
        Codepoint(char),
        Break,
    }

    /// A test case is composed of a sequence of codepoints and break points, along with an optional
    /// comment for debugging (e.g. containing an explanation of the test case etc).
    pub struct TestCase {
        pub sequence: Vec<SequenceItem>,
        pub comment: Option<String>,
    }

    impl TestCase {
        /// Converts the sequence of codepoints in the test case to a string, ignoring break points.
        /// The tokenizer's job is to re-insert these break points correctly.
        pub fn codepoints_as_string(&self) -> String {
            self.sequence
                .iter()
                .filter_map(|item| match item {
                    SequenceItem::Codepoint(c) => Some(*c),
                    SequenceItem::Break => None,
                })
                .collect()
        }
    }

    /// Loads the test cases from a UAX #29 break test file, parsing the codepoints, break points,
    /// and comments.
    pub fn load_break_tests(filepath: &str) -> Vec<TestCase> {
        let f = File::open(filepath).unwrap();
        let reader = BufReader::new(f);
        let mut tests = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            let mut trimmed_line = line.trim();
            if trimmed_line.is_empty() || trimmed_line.starts_with('#') {
                continue; // Skip empty lines and comments
            }
            let comment = if let Some(comment_start_idx) = trimmed_line.find('#') {
                let (before, after) = trimmed_line.split_at(comment_start_idx);
                trimmed_line = before.trim();
                let comment_str = after[1..].trim();
                (!comment_str.is_empty()).then(|| comment_str.to_string())
            } else {
                None
            };
            let mut sequence = Vec::new();
            for part in trimmed_line.split(' ') {
                if part == "÷" {
                    sequence.push(SequenceItem::Break);
                } else if part == "×" {
                    continue; // no break, ignore
                } else {
                    let codepoint = u32::from_str_radix(part, 16).unwrap();
                    let character = std::char::from_u32(codepoint).unwrap();
                    sequence.push(SequenceItem::Codepoint(character));
                }
            }
            tests.push(TestCase { sequence, comment });
        }
        tests
    }

    pub fn test_against_uax29_break_tests(
        test_filepath: &str,
        tokenize: impl Fn(&str, &mut Vec<usize>),
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut failures = Vec::new();
        let test_cases = load_break_tests(test_filepath);
        let mut breakpoints = Vec::new();
        for t in test_cases {
            let input_string = t.codepoints_as_string();
            tokenize(&input_string, &mut breakpoints);
            let mut got_sequence = Vec::new();
            input_string
                .char_indices()
                .merge_join_by(breakpoints.drain(..), |&(idx, _c), break_idx| {
                    idx.cmp(break_idx)
                })
                .for_each(|eob| match eob {
                    EitherOrBoth::Left((_, c)) => got_sequence.push(SequenceItem::Codepoint(c)),
                    EitherOrBoth::Right(_) => got_sequence.push(SequenceItem::Break),
                    EitherOrBoth::Both((_, c), _) => {
                        got_sequence.push(SequenceItem::Break);
                        got_sequence.push(SequenceItem::Codepoint(c));
                    }
                });
            if got_sequence == t.sequence {
                passed += 1;
            } else {
                failures.push((t, got_sequence));
            }
        }
        for failure in failures.iter() {
            let (expected, got) = failure;
            println!("expected: {:?}", expected.sequence);
            println!("     got: {:?}", got);
            if let Some(comment) = expected.comment.as_ref() {
                println!(" comment: {}", comment);
            }
        }
        (passed, failures.len())
    }
}
