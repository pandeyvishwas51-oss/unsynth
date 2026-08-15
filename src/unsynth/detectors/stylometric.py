"""Stylometric detector aimed at GPTZero / Originality-style feature stacks.

Features are intentionally inspectable: a researcher should be able to see
*why* a document looks machine-like, not just a black-box percentage.
"""

from __future__ import annotations

import math
from collections import Counter

from unsynth.data.lexicon import FUNCTION_WORD_WEIGHTS
from unsynth.detectors.base import BaseDetector
from unsynth.text import (
    FUNCTION_WORDS,
    clamp,
    hapax_ratio,
    honores_r,
    inv_logistic,
    logistic,
    lower_words,
    mean_std,
    ngrams,
    sentences,
    syllable_count,
    tokenize,
    type_token_ratio,
    words,
    yules_k,
)
from unsynth.types import DetectionContext, DetectorFamily, DetectorResult, Signal


class StylometricDetector(BaseDetector):
    name = "stylometric"
    family = DetectorFamily.STYLOMETRIC
    version = "1.0.0"

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        del context
        raw = text.strip()
        wds = words(raw)
        if len(wds) < 12:
            return self.empty_result(self.name, self.family, "text too short for stylometry")

        sents = sentences(raw)
        toks = tokenize(raw)
        lowers = [w.lower() for w in wds]
        sent_lens = [len(words(s)) for s in sents] or [len(wds)]
        word_lens = [len(w) for w in wds]
        punct = [t.text for t in toks if t.kind == "punct"]

        ttr = type_token_ratio(lowers)
        hapax = hapax_ratio(lowers)
        yule = yules_k(lowers)
        honore = honores_r(lowers)
        mean_sl, std_sl = mean_std([float(x) for x in sent_lens])
        mean_wl, _std_wl = mean_std([float(x) for x in word_lens])
        fw_chi = self._function_word_chi(lowers)
        comma_ratio = sum(1 for p in punct if p == ",") / max(1, len(wds))
        quote_ratio = raw.count('"') + raw.count("“") + raw.count("’s")
        contraction_ratio = sum(1 for w in wds if "'" in w) / max(1, len(wds))
        digit_ratio = sum(1 for t in toks if t.kind == "number") / max(1, len(toks))
        avg_syll = sum(syllable_count(w) for w in wds) / max(1, len(wds))
        start_diversity = self._sentence_start_diversity(sents)
        bigram_ent = self._ngram_entropy(lowers, 2)
        adj_sim = self._adjacent_sentence_similarity(sents)

        # Machine drafts: mid sentence length ~18–24, low start diversity,
        # few contractions, even adjacent sentences, high function-word fit
        # to "generic English essay" rather than an authorial fingerprint.
        signals = [
            Signal("type_token_ratio", ttr, 0.10),
            Signal("yules_k", yule, 0.08, "higher = more repetitive"),
            Signal("honores_r", honore, 0.05),
            Signal("mean_sentence_len", mean_sl, 0.08),
            Signal("sentence_len_std", std_sl, 0.10),
            Signal("function_word_chi", fw_chi, 0.10, "fit to generic English closed-class"),
            Signal("contraction_ratio", contraction_ratio, 0.10),
            Signal("sentence_start_diversity", start_diversity, 0.10),
            Signal("adjacent_sentence_sim", adj_sim, 0.10),
            Signal("bigram_entropy", bigram_ent, 0.07),
            Signal("comma_ratio", comma_ratio, 0.04),
            Signal("avg_syllables", avg_syll, 0.04),
            Signal("hapax_ratio", hapax, 0.04),
        ]

        score = (
            0.10 * inv_logistic(ttr, 0.55, 8.0)
            + 0.08 * logistic(yule, 120.0, 0.03)
            + 0.05 * inv_logistic(honore, 800.0, 0.004)
            + 0.08 * self._target_band(mean_sl, 16.0, 24.0)
            + 0.10 * inv_logistic(std_sl, 6.5, 0.25)
            + 0.10 * logistic(fw_chi, 0.55, 6.0)
            + 0.10 * inv_logistic(contraction_ratio, 0.025, 40.0)
            + 0.10 * inv_logistic(start_diversity, 0.62, 6.0)
            + 0.10 * logistic(adj_sim, 0.28, 8.0)
            + 0.07 * inv_logistic(bigram_ent, 4.2, 1.2)
            + 0.04 * self._target_band(comma_ratio, 0.04, 0.09)
            + 0.04 * self._target_band(avg_syll, 1.45, 1.75)
            + 0.04 * inv_logistic(hapax, 0.48, 6.0)
        )

        n = len(wds)
        confidence = clamp(0.40 + 0.40 * math.log10(max(n, 10)) / 3.0)
        return self.result(
            score,
            confidence=confidence,
            details={
                "tokens": n,
                "sentences": len(sents),
                "mean_sentence_len": round(mean_sl, 2),
                "sentence_len_std": round(std_sl, 2),
                "mean_word_len": round(mean_wl, 2),
                "ttr": round(ttr, 4),
                "yules_k": round(yule, 2),
                "contraction_ratio": round(contraction_ratio, 4),
                "start_diversity": round(start_diversity, 4),
                "digit_ratio": round(digit_ratio, 4),
                "quote_chars": quote_ratio,
            },
            signals=signals,
        )

    def _function_word_chi(self, tokens: list[str]) -> float:
        """How close is the closed-class distribution to generic English?

        Authorial writing leaves a fingerprint. Generic LLM essays hug the
        population frequencies of *the/of/and/to*.
        """

        if not tokens:
            return 0.0
        counts = Counter(tokens)
        n = len(tokens)
        chi = 0.0
        used = 0
        for word, expected_p in FUNCTION_WORD_WEIGHTS.items():
            observed = counts.get(word, 0) / n
            # Soft closeness: 1 when observed ≈ expected.
            denom = expected_p + 1e-6
            chi += 1.0 - min(1.0, abs(observed - expected_p) / denom)
            used += 1
        # Also penalize missing rare function words that humans still use.
        rare = [w for w in ("though", "yet", "nor", "amongst", "via") if w in FUNCTION_WORDS]
        missing = sum(1 for w in rare if counts.get(w, 0) == 0)
        closeness = chi / max(1, used)
        return clamp(0.75 * closeness + 0.25 * (missing / max(1, len(rare))))

    def _sentence_start_diversity(self, sents: list[str]) -> float:
        if not sents:
            return 0.0
        starts: list[str] = []
        for sent in sents:
            ws = lower_words(sent)
            if ws:
                starts.append(ws[0])
        if not starts:
            return 0.0
        return len(set(starts)) / len(starts)

    def _ngram_entropy(self, tokens: list[str], n: int) -> float:
        grams = ngrams(tokens, n)
        if not grams:
            return 0.0
        counts = Counter(grams)
        total = sum(counts.values())
        return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())

    def _adjacent_sentence_similarity(self, sents: list[str]) -> float:
        if len(sents) < 2:
            return 0.0
        scores: list[float] = []
        prev = set(lower_words(sents[0]))
        for sent in sents[1:]:
            cur = set(lower_words(sent))
            if not prev or not cur:
                scores.append(0.0)
            else:
                scores.append(len(prev & cur) / len(prev | cur))
            prev = cur
        return sum(scores) / len(scores)

    @staticmethod
    def _target_band(value: float, lo: float, hi: float) -> float:
        """1.0 if value sits inside the stereotypical-AI band."""

        if lo <= value <= hi:
            return 1.0
        width = max(hi - lo, 1e-6)
        if value < lo:
            return clamp(1.0 - (lo - value) / width)
        return clamp(1.0 - (value - hi) / width)
