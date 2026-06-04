# Changelog

## July 26, 2026

- `uax29::sentence`: much faster tokenization on ASCII-heavy text. The scanner now skips the
  per-byte property and transition-table loads for the interior of a sentence, classifying a machine
  word at a time and stopping only at bytes the DFA must actually inspect. Output is byte-for-byte
  identical — purely a throughput change.
- Corrected the declared MSRV to 1.88. This is not a bump: the crate has required 1.88 since it
  began using let-chains, and the manifests understated it as 1.85. CI now builds on exactly the
  declared floor.

## June 18, 2026

- `analyze`: add `Token::byte_range` and `Token::input_index` to recover a token's raw source substring

## May 8, 2026

- `analyze` module, support for stopword removal, stemming, ascii_folding, maximum_token_length, case sensitivity

## Apr 14, 2026

- Initial crate release
- UAX #29 compliant word and sentence tokenizer