# Changelog

All notable changes to UnSynth are documented here.

## 0.1.1 — 2026-08-15

### Added

- Production safety layer: input sanitization, finite score guards, URL/email
  span protection, paragraph chunking for long prose.
- Unicode-aware tokenizer (café / naïve / CJK / Cyrillic).
- Stress / fuzz / invariant suite (114 tests) plus `scripts/stress_bench.py`.

### Fixed

- `--json` no longer swallowed `--output` on `clean` / `rewrite`.
- Negative RNG seeds crashed hashing.
- Code fences with infostrings (` ```python:3.11 `) and `~~~` were rewritten.
- Phrase substitution could nibble inside URLs and compound words.

## 0.1.0 — 2026-08-15

### Added

- Detector plugin framework with classical (perplexity / burstiness / stock-phrase),
  stylometric, statistical watermark heuristics, and an Anthropic Detection API adapter.
- Multi-strategy rewriter stack: lexical, structural, style humanizer, optional
  local-LLM paraphrase and back-translation.
- Adaptive Detect → Rewrite → Re-detect pipeline with quality gates.
- Markdown-aware document handling (code fences, tables, front matter protected).
- C2PA / XMP / EXIF / HTML provenance stripper.
- CLI: `detect`, `rewrite`, `clean`, `strip`, `eval`, `batch`, `doctor`, `serve`.
- Evaluation harness and example notebook.
