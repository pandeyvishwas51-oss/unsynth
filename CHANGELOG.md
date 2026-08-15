# Changelog

All notable changes to UnSynth are documented here.

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
