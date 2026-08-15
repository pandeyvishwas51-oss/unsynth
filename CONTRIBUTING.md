# Contributing to UnSynth

Thanks for helping build a toolkit people can actually trust.

## Ground rules

1. **Honesty first.** Never claim cryptographic or 100% undetectable removal.
   If you add a detector, document what it *cannot* see.
2. **Local-first.** New network calls must be opt-in (explicit flag or env var).
3. **Plugins, not forks.** New detector families and rewrite strategies should
   subclass `BaseDetector` / `BaseRewriter` and register via entry points.
4. **Tests before adjectives.** A "robust" detector without a test that
   separates AI-flavored prose from messy human notes will be rejected.

## Dev setup

```bash
uv sync --extra dev
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src/unsynth
uv run pytest --cov
```

Python 3.11+ is required. 3.12/3.13 are CI-tested.

## Adding a detector

1. Subclass `unsynth.detectors.base.BaseDetector`.
2. Set `name`, `family`, `version`.
3. Return a `DetectorResult` with inspectable `signals` — not just a float.
4. Register the class in
   `[project.entry-points."unsynth.detectors"]` (in-tree) **or** ship a
   third-party package that exports the same group.
5. Add a test that scores two fixtures in the expected direction.

Watermark detectors belong in `DetectorFamily.WATERMARK` or `API`.
Do not fold a keyed vendor score into the classical average without labeling it.

## Adding a rewriter

1. Subclass `unsynth.rewriters.base.BaseRewriter`.
2. Honor `strength ∈ [0, 1]`.
3. Leave protected markdown segments alone — the orchestrator handles that,
   but do not invent your own document parser unless you need to.
4. If you need an LLM, go through `unsynth.backends.complete` so cloud
   stays optional.

## PR checklist

- [ ] Types + ruff clean
- [ ] Tests for the new path
- [ ] Limitations documented if you touch detection or watermark claims
- [ ] No secrets, no sample text that is someone else's unpublished draft

## Release

Versions follow SemVer. `0.x` may break APIs; we will call it out in
`CHANGELOG.md`.
