# Changelog

## June 20, 2026

- `uax29::word`: much faster ASCII word tokenization. The hot scan now uses SIMD backends (SSE2 or,
  on an SSSE3 build, a PSHUFB classifier on x86-64; NEON on aarch64; portable SWAR elsewhere), and a
  fused fast-path emits the common `word`/space alternation without re-entering the DFA. Build with
  `-C target-cpu=x86-64-v2` (or `+ssse3`) to select the PSHUFB path on x86-64; the default build is
  unchanged and equally correct. Output is byte-for-byte identical — purely a throughput change.

## June 18, 2026

- `analyze`: add `Token::byte_range` and `Token::input_index` to recover a token's raw source substring

## May 8, 2026

- `analyze` module, support for stopword removal, stemming, ascii_folding, maximum_token_length, case sensitivity

## Apr 14, 2026

- Initial crate release
- UAX #29 compliant word and sentence tokenizer