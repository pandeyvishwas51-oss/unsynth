"""Quality gates: semantic similarity, readability, length, token change."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from unsynth.config import Settings
from unsynth.logging import get_logger
from unsynth.safety import finite_unit
from unsynth.text import (
    char_ngrams,
    cosine_counters,
    flesch_reading_ease,
    jaccard,
    lower_words,
    tfidf_cosine,
    token_change_rate,
)
from unsynth.types import QualityReport, TokenChangeStats

log = get_logger("quality")

_EMBEDDER: Any = None
_EMBEDDER_FAILED = False


def _try_embedder(model_name: str) -> Any:
    global _EMBEDDER, _EMBEDDER_FAILED
    if _EMBEDDER is not None or _EMBEDDER_FAILED:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
        log.info("loaded sentence-transformers model %s", model_name)
    except Exception as exc:  # optional extra
        _EMBEDDER_FAILED = True
        log.debug("sentence-transformers unavailable: %s", exc)
        _EMBEDDER = None
    return _EMBEDDER


@lru_cache(maxsize=128)
def _embed_cached(model_name: str, text: str) -> tuple[float, ...] | None:
    model = _try_embedder(model_name)
    if model is None:
        return None
    vec = model.encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vec)


def embedding_cosine(a: str, b: str, model_name: str) -> float | None:
    va = _embed_cached(model_name, a)
    vb = _embed_cached(model_name, b)
    if va is None or vb is None:
        return None
    return sum(x * y for x, y in zip(va, vb, strict=True))


class QualityGate:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def similarity(self, original: str, rewritten: str) -> tuple[float, str]:
        mode = self.settings.quality.embeddings
        if mode in {"auto", "sentence-transformers"}:
            cos = embedding_cosine(original, rewritten, self.settings.quality.embedding_model)
            if cos is not None:
                return float(cos), "sentence-transformers"
            if mode == "sentence-transformers":
                log.warning("sentence-transformers requested but unavailable; using tfidf")
        return tfidf_cosine(original, rewritten), "tfidf"

    def evaluate(self, original: str, rewritten: str) -> QualityReport:
        sim, method = self.similarity(original, rewritten)
        sim = finite_unit(sim)
        rb = flesch_reading_ease(original)
        ra = flesch_reading_ease(rewritten)
        o_len = max(1, len(original.strip()))
        r_len = len(rewritten.strip())
        ratio = r_len / o_len
        change = token_change_rate(original, rewritten)
        reasons: list[str] = []
        q = self.settings.quality
        rw = self.settings.rewrite
        if sim < rw.min_similarity:
            reasons.append(f"similarity {sim:.3f} < min {rw.min_similarity:.3f}")
        if ratio > q.max_length_ratio:
            reasons.append(f"length ratio {ratio:.2f} > {q.max_length_ratio:.2f}")
        if ratio < q.min_length_ratio:
            reasons.append(f"length ratio {ratio:.2f} < {q.min_length_ratio:.2f}")
        if ra < q.min_readability and ra + 15.0 < rb:
            reasons.append(f"readability collapsed ({rb:.1f} → {ra:.1f})")
        if not rewritten.strip():
            reasons.append("empty rewrite")
        return QualityReport(
            similarity=sim,
            readability_before=rb,
            readability_after=ra,
            length_ratio=ratio,
            token_change_rate=change,
            passed=not reasons,
            reasons=tuple(reasons),
            method=method,
        )

    def token_stats(self, original: str, rewritten: str) -> TokenChangeStats:
        a = lower_words(original)
        b = lower_words(rewritten)
        change = token_change_rate(original, rewritten)
        return TokenChangeStats(
            original_tokens=len(a),
            rewritten_tokens=len(b),
            changed_tokens=int(round(change * max(len(a), len(b)))),
            change_rate=change,
            unigram_jaccard=jaccard(a, b),
            char_ngram_cosine=cosine_counters(char_ngrams(original), char_ngrams(rewritten)),
        )
