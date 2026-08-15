"""Classical AI-text detector (perplexity + burstiness + ZeroGPT-style cues).

This is *not* a watermark detector. It approximates the public classifiers
people actually hit first: ZeroGPT, GPTZero, Originality, Turnitin's
AI report, etc. Those systems are stylometric / LM-perplexity ensembles.

Without a local language model we estimate "perplexity" from unigram Zipf
deviation + local n-gram reuse. When a backend *is* configured, the
optional LM scorer in ``unsynth.backends`` can replace that estimate.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from unsynth.data.lexicon import AI_OPENERS, AI_PHRASES, unigram_logprob
from unsynth.detectors.base import BaseDetector
from unsynth.text import (
    FUNCTION_WORDS,
    burstiness,
    clamp,
    hapax_ratio,
    inv_logistic,
    logistic,
    lower_words,
    mean_std,
    ngrams,
    sentences,
    tokenize,
    type_token_ratio,
    words,
)
from unsynth.types import DetectionContext, DetectorFamily, DetectorResult, Signal

_AI_WORD_RE = re.compile(
    r"\b(delve|leverage|utilize|robust|comprehensive|cutting[- ]edge|"
    r"landscape|tapestry|paradigm|holistic|streamline|empower|harness|"
    r"groundbreaking|seamless|unlock|foster|optimize|revolutionize|"
    r"furthermore|moreover|additionally|consequently|subsequently|"
    r"testament|underscores|pivotal|multifaceted|plethora|myriad)\b",
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r"\b(it is important to note|it's important to note|in conclusion|"
    r"in summary|in today's world|in today's digital|plays a (crucial|vital|significant) role|"
    r"a wide range of|in order to|due to the fact that|at the end of the day|"
    r"when it comes to|needless to say|first and foremost|last but not least)\b",
    re.IGNORECASE,
)


class ClassicalDetector(BaseDetector):
    name = "classical"
    family = DetectorFamily.CLASSICAL
    version = "1.0.0"

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        del context
        raw = text.strip()
        if len(words(raw)) < 12:
            return self.empty_result(self.name, self.family, "text too short for a stable score")

        sents = sentences(raw)
        toks = tokenize(raw)
        wds = [t.text for t in toks if t.is_word]
        lowers = [w.lower() for w in wds]
        sent_lens = [len(words(s)) for s in sents] or [len(wds)]

        burst = burstiness(sent_lens)
        ttr = type_token_ratio(lowers)
        hapax = hapax_ratio(lowers)
        ppl = self._approx_perplexity(lowers)
        punct_var = self._punctuation_uniformity(raw)
        phrase_hit = self._phrase_density(raw)
        opener_hit = self._opener_score(sents)
        length_uniform = self._length_uniformity(sent_lens)
        reuse = self._ngram_reuse(lowers)
        function_ratio = sum(1 for w in lowers if w in FUNCTION_WORDS) / max(1, len(lowers))

        # Higher score = more AI-like.
        # Human text: high burstiness, higher TTR/hapax, less template phrasing.
        signals = [
            Signal("approx_perplexity", ppl, 0.22, "lower ≈ more predictable / template-like"),
            Signal("burstiness", burst, 0.16, "low sentence-length variance is an AI tell"),
            Signal("type_token_ratio", ttr, 0.10, "AI drafts often look lexically flat"),
            Signal("hapax_ratio", hapax, 0.06, "share of once-only words"),
            Signal("phrase_density", phrase_hit, 0.16, "ZeroGPT-style stock phrases"),
            Signal("opener_templates", opener_hit, 0.08, "In today's / In conclusion openers"),
            Signal("length_uniformity", length_uniform, 0.08, "too-even sentence lengths"),
            Signal("ngram_reuse", reuse, 0.08, "repeated 3-grams look templated"),
            Signal("punctuation_uniformity", punct_var, 0.04, "predictable comma/period rhythm"),
            Signal("function_word_ratio", function_ratio, 0.02, "closed-class density"),
        ]

        mix = (
            0.14 * inv_logistic(ppl, midpoint=80.0, steepness=0.04)
            + 0.12 * inv_logistic(burst, midpoint=0.55, steepness=5.0)
            + 0.08 * inv_logistic(ttr, midpoint=0.52, steepness=8.0)
            + 0.04 * inv_logistic(hapax, midpoint=0.45, steepness=6.0)
            + 0.30 * clamp(phrase_hit)
            + 0.12 * clamp(opener_hit)
            + 0.08 * clamp(length_uniform)
            + 0.06 * clamp(reuse)
            + 0.04 * clamp(punct_var)
            + 0.02 * logistic(function_ratio, midpoint=0.45, steepness=10.0)
        )
        # Stock-phrase density is the ZeroGPT-style tell. Don't let a
        # reasonably bursty sentence rhythm wash it out.
        if phrase_hit >= 0.55:
            mix = max(mix, 0.52 + 0.40 * phrase_hit)
        score = mix

        # Short docs are noisy.
        n = len(wds)
        confidence = clamp(0.35 + 0.45 * math.log10(max(n, 10)) / 3.0)
        if n < 80:
            confidence *= 0.7

        return self.result(
            score,
            confidence=confidence,
            details={
                "tokens": n,
                "sentences": len(sents),
                "approx_perplexity": round(ppl, 2),
                "burstiness": round(burst, 4),
                "type_token_ratio": round(ttr, 4),
                "phrase_hits": round(phrase_hit, 4),
            },
            signals=signals,
        )

    def _approx_perplexity(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        # Mean unigram surprisal, scaled toward a GPT-2-like PPL range.
        surprisal = [-unigram_logprob(t) for t in tokens]
        mean_nats = sum(surprisal) / len(surprisal)
        return float(math.exp(mean_nats))

    def _phrase_density(self, text: str) -> float:
        hits = len(_HEDGE_RE.findall(text)) + len(_AI_WORD_RE.findall(text))
        # Also count multi-word lexicon keys.
        blob = text.lower()
        for phrase in AI_PHRASES:
            if " " in phrase and phrase in blob:
                hits += 1
        tokens = max(1, len(lower_words(text)))
        # 1 hit / 80 tokens is already quite "AI-flavored".
        return clamp(hits / max(1.0, tokens / 80.0) / 3.0 * 1.4)

    def _opener_score(self, sents: list[str]) -> float:
        if not sents:
            return 0.0
        hits = 0
        checked = sents[:8] + sents[-3:]
        for sent in checked:
            head = sent.lower()[:48]
            if any(head.startswith(op) or op in head[:40] for op in AI_OPENERS):
                hits += 1
        return clamp(hits / max(2.0, len(checked) / 3.0))

    def _length_uniformity(self, lengths: list[int]) -> float:
        if len(lengths) < 3:
            return 0.0
        mean, std = mean_std([float(x) for x in lengths])
        if mean <= 0:
            return 0.0
        cv = std / mean
        # Human CV often 0.4–0.8; many LLM drafts sit near 0.2–0.35.
        return inv_logistic(cv, midpoint=0.38, steepness=8.0)

    def _ngram_reuse(self, tokens: list[str]) -> float:
        tris = ngrams(tokens, 3)
        if len(tris) < 12:
            return 0.0
        counts = Counter(tris)
        reused = sum(c - 1 for c in counts.values() if c > 1)
        return clamp(reused / len(tris) * 4.0)

    def _punctuation_uniformity(self, text: str) -> float:
        marks = [ch for ch in text if ch in ",.;:!?—-"]
        if len(marks) < 8:
            return 0.3
        counts = Counter(marks)
        # Entropy of punct types; low entropy = robotic comma/period only.
        total = sum(counts.values())
        ent = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
        max_ent = math.log(max(2, len(counts)))
        if max_ent <= 0:
            return 0.3
        return inv_logistic(ent / max_ent, midpoint=0.55, steepness=5.0)
