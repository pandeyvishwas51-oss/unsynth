# UnSynth

**Local-first toolkit for detecting AI-generated text and disrupting classical detectors *and* statistical LLM watermarks** (SynthID-Text / tournament-sampling / Kirchenbauer-style), including text that is already published.

[![CI](https://img.shields.io/github/actions/workflow/status/pandeyvishwas51-oss/unsynth/ci.yml?branch=main)](https://github.com/pandeyvishwas51-oss/unsynth/actions)
[![PyPI](https://img.shields.io/pypi/v/unsynth.svg)](https://pypi.org/project/unsynth/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Local-first](https://img.shields.io/badge/default-Ollama%20%2F%20offline-informational.svg)](#language-model-backends)

> **Honesty first.** UnSynth does **not** cryptographically erase a private-key generative watermark and does **not** make text “100% undetectable.” Those claims are how the last generation of “humanizer” tools burned their users. What UnSynth *does* is the strongest practical thing you can do without the secret key: change the token sequence and the local context a watermark is seeded from, inject human style variance, and re-score until the public detectors go quiet — or tell you they didn’t.

Paste a blog post. Get a detection report. Get a cleaned draft. Keep the code blocks.

```bash
pip install unsynth
# or
uv add unsynth

unsynth detect article.md
unsynth clean  article.md -o article.clean.md --report article.report.md
unsynth strip  photo.jpg --inplace
```

---

## Why this exists

Two different things get sold as “AI detection”:

| | Classical detectors | Generative watermarks |
|---|---|---|
| Examples | ZeroGPT, GPTZero, Originality, Turnitin AI, CopyLeaks | Kirchenbauer et al. 2023, [SynthID-Text](https://deepmind.google/technologies/synthid/) (Google DeepMind), reported Claude / Anthropic watermarking |
| Signal | Style, burstiness, perplexity, stock phrasing | Secret-key hash of *local context* biasing the next token |
| Strength | Brittle, easy to overfit, high false positives on formal writing | Strong *if you have the key*; weak to paraphrase and context edits |
| What fools it | Contractions, bursty sentences, synonym / structure change | **Changing tokens**, especially in high-entropy positions, which also reseeds every later hash |

Most open-source “humanizers” only fight the left column, and they fight it by calling a cloud LLM. UnSynth treats the right column as a first-class target, stays **offline by default**, and is built so the next watermark family is a plugin, not a rewrite.

### Keywords this repo is actually about

Claude watermark, SynthID-Text, SynthID removal, tournament sampling, g-values, Kirchenbauer watermark, AI detection bypass, humanize AI text, GPTZero / ZeroGPT / Originality, C2PA / Content Credentials strip, local-first AI text cleaner.

---

## How SynthID-Text / Claude-style watermarking actually works

This is the part most READMEs hand-wave. The details matter because they dictate *what a remover is allowed to claim*.

### 1. Classical logit watermarks (Kirchenbauer, “green/red list”)

At step \(t\) the model is about to sample \(x_t\) from logits \(\ell\) over the vocabulary.

1. Hash the previous token(s) together with a **secret key** \(k\).
2. Use that hash to partition the vocabulary into a *green* list of fraction \(\gamma\) and a *red* list.
3. Add a bias \(\delta\) to every green logit, then sample.

A keyed detector counts how often the emitted token landed in green and computes a z-score against \(\mathrm{Binomial}(n, \gamma)\). Human text sits near 0.5. Watermarked text sits high — **but only if you know \(k\)**.

### 2. Tournament sampling (SynthID-Text)

[SynthID-Text](https://arxiv.org/abs/2305.13678) (DeepMind; also the public description closest to what Anthropic has discussed for Claude) does not boost a green list. It runs a **tournament**:

1. Draw \(m\) candidate tokens from the model.
2. For each candidate \(v\), compute one or more **g-values**
   \(g_i = H(k, \text{context}, v, i) \in \{0,1\}\)
   from a keyed hash of the *context window* plus the candidate.
3. Pair candidates; the one with the higher g-value wins. Repeat until one token remains.
4. Detection is a hypothesis test on the **mean g-value** of the generated sequence (multiple g-functions give a hierarchical / variable-confidence test).

Two properties fall out immediately:

- **The watermark lives in the token identities and their local context, not in metadata.** Stripping C2PA from a Word doc does nothing to a SynthID-Text span.
- **High-entropy positions carry most of the signal.** When the model was already sure (`the`, `of`, `to`) the tournament has nothing to decide. When the model had a real choice (`robust` vs `sturdy` vs `solid`) the hash gets a vote. That is why UnSynth’s rewriter ranks tokens by an entropy prior and prefers to touch those.
- **Edits cascade.** Changing \(x_t\) changes the context hash for \(x_{t+1}, x_{t+2}, \ldots\). A few well-placed substitutions destroy more watermark alignment than a blanket thesaurus pass. Structural edits (split / merge / clause flip) destroy n-gram windows even harder.

### 3. What UnSynth can detect without the key

**Not a courtroom detector.** The `statistical` module is a *blind* heuristic:

- multi-seed green-list scans (max z-score and z-spread across random partitions);
- context-hash consistency (sticky continuations of a bigram);
- local window spikes, used as rewrite targets;
- optional rank/entropy gap when a local LM backend is configured.

A high statistical score is a **hint to rewrite that span**, not a proof the text is watermarked. The `anthropic` detector is a ready-to-plug adapter for a future keyed Detection API; it stays silent unless you set `UNSYNTH_ANTHROPIC_API_KEY` and `UNSYNTH_ANTHROPIC_DETECTION_URL`. UnSynth will not invent a “Claude watermark found” number from local math.

---

## What UnSynth is (and is not)

**Is**

- A modular detect + disrupt toolkit for *existing* articles, posts, and documents.
- A research-grade evaluation harness (before/after scores, similarity, readability, token change).
- A C2PA / XMP / EXIF / HTML provenance stripper.
- Plugin-shaped: new detectors and rewriters register via setuptools entry points.

**Is not**

- A way to “remove” a watermark the way you remove EXIF. Statistical watermarks are not metadata.
- A guarantee against a vendor who holds the secret key and a long enough residual span.
- A cheating service. If you are trying to submit AI work as human under a class or publisher policy, stop. This repo is for researchers, journalists verifying their own drafts, and engineers testing watermark robustness.

---

## Install

```bash
# core (heuristics, CLI, no GPU)
pip install unsynth

# recommended extras
pip install 'unsynth[embeddings]'   # sentence-transformers quality gate
pip install 'unsynth[llm]'          # Ollama client extras
pip install 'unsynth[metadata]'     # Pillow / pypdf / lxml
pip install 'unsynth[web]'          # local FastAPI UI
pip install 'unsynth[all]'
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add unsynth
uv sync --extra dev
```

Python **3.11+**. No account, no API key, no telemetry.

---

## Quickstart

### 1. Detect an already-published article

```bash
unsynth detect path/to/post.md
# or
cat post.md | unsynth detect --json
```

```text
          UnSynth detect
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━┓
┃ detector     ┃ family      ┃ score ┃ label     ┃ conf ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│ classical    │ classical   │ 0.81  │ ai        │ 0.66 │
│ stylometric  │ stylometric │ 0.74  │ ai        │ 0.61 │
│ statistical  │ watermark   │ 0.41  │ uncertain │ 0.33 │
│ ensemble     │ ensemble    │ 0.72  │ ai        │ 0.58 │
└──────────────┴─────────────┴───────┴───────────┴──────┘
```

### 2. Clean it (detect → rewrite → re-detect)

```bash
unsynth clean post.md -o post.clean.md --report post.report.md
```

The orchestrator:

1. Scores the original with the ensemble.
2. Rewrites **only prose** (code fences, tables, front matter stay put).
3. Prefers high-entropy tokens (watermark-bearing positions).
4. Re-scores. If classical or watermark heuristics are still high, it **raises strength and switches pressure** for another pass.
5. Rejects any pass that drops embedding / TF-IDF similarity below `min_similarity` (default 0.82).

### 3. Strip provenance metadata from a file

```bash
unsynth strip essay.pdf -o essay.stripped.pdf
unsynth strip hero.png --inplace
unsynth strip index.html
```

Removes C2PA markers, XMP packets, EXIF, and common HTML generator / Content Credentials tags. This is complementary to text cleaning — it does not touch the words.

### 4. From Python

```python
from pathlib import Path

from unsynth import UnSynthPipeline, load_settings
from unsynth.types import PipelineMode

text = Path("post.md").read_text()
pipe = UnSynthPipeline(load_settings())
print(pipe.detect(text).as_dict())

result = pipe.run(text, mode=PipelineMode.CLEAN)
assert result.after and result.eval.quality
print(result.before.score, "→", result.after.score, "sim", result.eval.quality.similarity)
Path("post.clean.md").write_text(result.output)
```

More: [`examples/detect_article.py`](examples/detect_article.py), [`examples/clean_published.py`](examples/clean_published.py), [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb).

---

## Architecture

```
src/unsynth/
  detectors/     classical · stylometric · statistical · anthropic adapter · ensemble
  rewriters/     lexical · structural · style · paraphrase · backtranslate · quality
  backends/      none | ollama | openai-compatible | transformers
  metadata/      C2PA / XMP / EXIF / HTML / PDF
  pipeline/      markdown-aware document model + adaptive orchestrator
  eval/          before/after reports
  cli/           typer + rich
  web/           optional FastAPI
```

Every detector implements `BaseDetector.detect(text) -> DetectorResult` with named `signals`. Every rewriter implements `BaseRewriter.rewrite(text, strength) -> RewriteResult` and goes through `QualityGate`. The orchestrator does not know how SynthID works; it only knows scores, families, and quality.

Community plugins:

```toml
[project.entry-points."unsynth.detectors"]
my_detector = "my_pkg.detectors:MyDetector"

[project.entry-points."unsynth.rewriters"]
my_rewriter = "my_pkg.rewriters:MyRewriter"
```

Or drop a `.py` file in a directory listed under `runtime.plugin_dirs`.

---

## Rewriter stack (what actually changes the tokens)

| Layer | Needs LLM? | Job |
|---|---|---|
| **lexical** | no | Stock-phrase kill list (`delve`, `leverage`, `in conclusion`…) + entropy-ranked synonym swaps |
| **structural** | no | Split coordinated sentences, merge shorts, flip *Although X, Y* → *Y, although X* — destroys n-gram context |
| **style** | no | Contractions, asides, burstiness, opener variance — aimed at GPTZero-style features |
| **paraphrase** | yes (local) | Multi-pass instruction paraphrase via Ollama / llama.cpp / transformers |
| **backtranslate** | yes, **off** | fr↔en pivot. Meaning-risky; enable with `rewrite.allow_backtranslate: true` |

Quality gate (always on):

- cosine similarity (sentence-transformers if installed, else word+char TF-IDF);
- Flesch readability floor;
- length ratio bounds;
- token-change statistics for the eval report.

---

## Language-model backends

Default is **`backend.kind: none`**. The heuristic stack is already useful.

```yaml
backend:
  kind: ollama          # none | ollama | openai_compatible | transformers
  model: llama3.1:8b
  host: http://127.0.0.1:11434
```

```bash
# local Ollama
ollama pull llama3.1:8b
UNSYNTH_BACKEND__KIND=ollama UNSYNTH_BACKEND__MODEL=llama3.1:8b unsynth clean post.md
```

An OpenAI-compatible URL is supported for people running vLLM / llama.cpp server — it is **never** the default and will not be called unless you set it. There is no hidden `api.openai.com` fallback.

---

## Configuration

Copy [`unsynth.example.yaml`](unsynth.example.yaml) to `unsynth.yaml` in the working directory, or to `~/.config/unsynth/config.yaml`. Environment variables use the `UNSYNTH_` prefix and `__` for nesting (`UNSYNTH_REWRITE__MAX_PASSES=6`).

Important knobs:

```yaml
rewrite:
  max_passes: 4
  min_similarity: 0.82          # raise this if meaning drift scares you
  target_ai_score: 0.40
  target_watermark_score: 0.48
  initial_strength: 0.42
  protect_markdown: true
  protect_code: true
  protect_tables: true
```

`unsynth doctor` prints what is actually wired up.

---

## CLI map

| Command | Purpose |
|---|---|
| `unsynth detect FILE` | Ensemble score, no edits |
| `unsynth rewrite FILE -s 0.6` | Single pass at a fixed strength |
| `unsynth clean FILE` | Adaptive detect/rewrite/re-detect |
| `unsynth strip FILE` | Provenance metadata |
| `unsynth eval ORIG CLEAN` | Research before/after |
| `unsynth batch DIR -o out/` | Directory tree |
| `unsynth doctor` | Backends, extras, plugins |
| `unsynth serve` | Local web UI (`unsynth[web]`) |

`--json` is available on the scoring commands. `--dry-run` on `rewrite` scores without writing.

---

## Limitations (read this)

1. **No secret key ⇒ no certified watermark removal.** A vendor detector with \(k\) and a long enough leftover span can still fire. UnSynth maximises disruption per unit of meaning change. That is the ceiling.
2. **Blind statistical detection is weak.** Do not cite `statistical` scores as evidence a document “is watermarked.” Use them to aim the rewriter.
3. **Classical detectors are a moving target.** ZeroGPT’s next prompt-injection will not match our phrase list. The plugin system is how we keep up; the phrase list is not a moat.
4. **Meaning vs. stealth is a tradeoff.** Crank `strength` and `max_passes` and you *will* drift facts. The similarity gate is a seatbelt, not a fact-checker. Read the output.
5. **Short texts are noisy.** Below ~80 tokens every detector here (and every public one) is guessing.
6. **Metadata stripping is best-effort.** Some PDFs restream XMP; some cameras write maker-notes UnSynth does not parse. Re-inspect with `exiftool` / `c2patool` if you need a forensic guarantee.
7. **This will not beat a human editor.** If the standard is “a suspicious professor reading carefully,” rewrite it yourself.

---

## Evaluation

```bash
uv run python scripts/evaluate.py examples/sample_article.md --json
uv run pytest --cov
```

The harness records ensemble scores before/after, family breakdowns, embedding similarity, Flesch, and token-level change rate. Please include those numbers in PRs that claim a rewriter is “stronger.”

---

## Project status

`0.1.0` is a research-grade foundation: the detector and rewriter cores, the adaptive pipeline, the metadata stripper, and the CLI. The Anthropic adapter is a contract, not a live integration. Official keyed APIs will land behind the same interface when they exist.

Roadmap we will actually accept PRs for:

- [ ] Token-level LM entropy when Ollama is up (replace the Zipf prior)
- [ ] Streaming paragraph cleaner for very long docs
- [ ] More watermark-family plugins (Unigram, NS-Watermark, Robust Distortion)
- [ ] Calibrated thresholds on a public human/LLM corpus
- [ ] Gradio frontend alongside FastAPI

---

## Citing / prior art

- Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023.
- Dathathri et al. / DeepMind, *SynthID-Text* (tournament sampling, g-values).
- Aaronson / OpenAI notes on cryptographic watermarks (different family; still token-level).
- GPTZero / ZeroGPT public writeups on burstiness + perplexity.

If you use UnSynth in a paper, please cite the repo URL and the version, and please do not write that it “removes” SynthID.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security: [SECURITY.md](SECURITY.md).

```bash
uv sync --extra dev
uv run ruff check src tests
uv run mypy src/unsynth
uv run pytest
```

## License

[MIT](LICENSE).
