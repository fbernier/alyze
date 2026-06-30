// Guards the field_match speedup: it scales with field length, so bench across field sizes.
// Synthetic, deterministic input — no corpus download needed.
use alyze::analyze::{AnalysisOptions, Analyzer, ReusableBuffer, TokenizerOptions};
use alyze_features::{Analyzed, TermStats, field_match};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn analyzer() -> Analyzer {
    Analyzer::new(AnalysisOptions {
        tokenizer: TokenizerOptions::UAX29Word(Default::default()),
        maximum_token_length: None,
        case_sensitive: false,
        stopword_removal: None,
        stemming: None,
        ascii_folding: false,
    })
}

/// `n` deterministic space-separated words. Query terms are 8 of a 512-word vocab, so they occur
/// *sparsely* (~n/512 each) — the realistic re-ranking regime. (A small all-query-term vocab would
/// instead make matches dense, which favours the old early-returning scan and isn't representative.)
fn doc_text(n: usize) -> String {
    use std::fmt::Write;
    let query = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    ];
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut s = String::new();
    for i in 0..n {
        if i > 0 {
            s.push(' ');
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let v = ((x >> 8) % 512) as usize;
        match query.get(v) {
            Some(w) => s.push_str(w),
            None => {
                let _ = write!(s, "w{v}");
            }
        }
    }
    s
}

fn bench_field_match(c: &mut Criterion) {
    let analyzer = analyzer();
    let mut buf = ReusableBuffer::new();
    let query = Analyzed::from_text(
        &analyzer,
        &mut buf,
        "the quick brown fox jumps over the lazy dog",
    );

    let mut group = c.benchmark_group("field_match");
    for &n in &[32usize, 256, 2048] {
        let doc = Analyzed::from_text(&analyzer, &mut buf, &doc_text(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &doc, |b, doc| {
            b.iter(|| field_match(&query, doc, |_| TermStats::default()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_field_match);
criterion_main!(benches);
