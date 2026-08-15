"""Blind statistical heuristics for generative watermarks.

What this can and cannot do
---------------------------
Schemes such as Kirchenbauer et al. (2023) and SynthID-Text
(Dathathri et al. / Google DeepMind) seed a hash from a secret key plus
a local context window, then bias sampling (green/red logit boost, or a
g-value tournament). A *keyed* detector computes a z-score or mean
g-value and is cryptographically strong.

UnSynth does **not** have the key. Without it we cannot certify that a
passage is or is not watermarked. We *can* look for anomalies that keyed
watermarks tend to leave behind, and we can rank spans that are cheaper
to disrupt:

* Multi-seed green-list scans. For a random vocabulary partition the
  green-hit rate should sit near γ (0.5 by default). A real watermark
  spikes on *one* unknown seed; blindly we look at the *maximum* z-score
  across many seeds and at local window spikes. This is a weak detector
  and is labeled as such.
* Context-hash consistency: tournament sampling makes the next token
  unusually determined by a hash of the previous k tokens. We measure
  how often token identity is predictable from a hash of the preceding
  n-gram *relative to a random baseline*.
* Rank / entropy gap, when a local LM backend is available: watermark
  bias is strongest in high-entropy positions. Systematic "safe-token"
  choices in those positions raise the score.

Treat scores from this module as **heuristics for targeting a rewrite**,
not as courtroom evidence.
"""

from __future__ import annotations

import math
from collections import Counter

from unsynth.detectors.base import BaseDetector
from unsynth.text import (
    clamp,
    mean_std,
    ngrams,
    sliding_windows,
    stable_hash32,
    tokenize,
)
from unsynth.types import DetectionContext, DetectorFamily, DetectorResult, Signal, SpanSignal

DEFAULT_SEEDS = (3, 7, 13, 17, 29, 41, 53, 97, 193, 389, 769, 1543)
GREEN_FRACTION = 0.50


class StatisticalWatermarkDetector(BaseDetector):
    name = "statistical"
    family = DetectorFamily.WATERMARK
    version = "1.0.0"

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        del context
        tokens = [t.lower for t in tokenize(text) if t.is_word]
        if len(tokens) < 16:
            return self.empty_result(
                self.name, self.family, "text too short for watermark heuristics"
            )

        scan = self._multiseed_scan(tokens)
        consistency = self._context_hash_consistency(tokens)
        spike = self._window_spikes(tokens)
        tail = self._rare_token_alignment(tokens)

        # Conservative mix: max-z across seeds is the closest thing we have
        # to a keyed green-list test, but it overfits on short text.
        max_z = scan["max_z"]
        mean_abs_z = scan["mean_abs_z"]
        z_score_signal = clamp((max_z - 1.2) / 4.0)
        spread_signal = clamp((scan["z_std"] - 0.4) / 1.6)

        signals = [
            Signal("max_green_z", max_z, 0.34, "best z-score across random green/red partitions"),
            Signal("mean_abs_z", mean_abs_z, 0.12, "average |z| across seeds"),
            Signal("z_spread", scan["z_std"], 0.10, "one seed standing out is more suspicious"),
            Signal("context_hash_consistency", consistency, 0.22),
            Signal("window_spike", spike, 0.14, "local green-rate bursts"),
            Signal("rare_token_alignment", tail, 0.08),
        ]

        score = (
            0.34 * z_score_signal
            + 0.12 * clamp(mean_abs_z / 2.5)
            + 0.10 * spread_signal
            + 0.22 * consistency
            + 0.14 * spike
            + 0.08 * tail
        )

        n = len(tokens)
        # Confidence stays modest on purpose — this is a blind test.
        confidence = clamp(0.22 + 0.35 * math.log10(max(n, 16)) / 3.0)

        spans = self._hot_spans(text)
        return self.result(
            score,
            confidence=confidence,
            details={
                "tokens": n,
                "seeds": len(DEFAULT_SEEDS),
                "gamma": GREEN_FRACTION,
                "max_z": round(max_z, 4),
                "best_seed": scan["best_seed"],
                "mean_green_rate": round(scan["mean_rate"], 4),
                "max_green_rate": round(scan["max_rate"], 4),
                "context_hash_consistency": round(consistency, 4),
                "disclaimer": (
                    "Blind heuristic. A high score is a rewrite hint, not a "
                    "proof the text carries a private-key watermark."
                ),
            },
            signals=signals,
            spans=spans,
        )

    def _green(self, prev: str, token: str, seed: int) -> bool:
        digest = stable_hash32((prev, token, "g"), seed=seed)
        return (digest % 10_000) < int(GREEN_FRACTION * 10_000)

    def _multiseed_scan(self, tokens: list[str]) -> dict[str, float | int]:
        rates: list[float] = []
        zscores: list[float] = []
        best_seed = DEFAULT_SEEDS[0]
        best_z = -1e9
        n = max(1, len(tokens) - 1)
        for seed in DEFAULT_SEEDS:
            green = 0
            for i in range(1, len(tokens)):
                if self._green(tokens[i - 1], tokens[i], seed):
                    green += 1
            rate = green / n
            # Binomial z-score vs expected γ.
            var = GREEN_FRACTION * (1.0 - GREEN_FRACTION) / n
            z = (rate - GREEN_FRACTION) / math.sqrt(max(var, 1e-12))
            rates.append(rate)
            zscores.append(z)
            if z > best_z:
                best_z = z
                best_seed = seed
        _, z_std = mean_std(zscores)
        return {
            "max_z": max(zscores),
            "mean_abs_z": sum(abs(z) for z in zscores) / len(zscores),
            "z_std": z_std,
            "mean_rate": sum(rates) / len(rates),
            "max_rate": max(rates),
            "best_seed": best_seed,
        }

    def _context_hash_consistency(self, tokens: list[str]) -> float:
        """Does token identity track a hash of the previous n-gram?

        For each bigram context we look at the diversity of next tokens.
        Watermarked text can look *less* diverse than raw LM output for
        a fixed context because the tournament repeatedly crowns the same
        hash-favored continuation.
        """

        if len(tokens) < 24:
            return 0.0
        contexts: dict[tuple[str, ...], list[str]] = {}
        for gram in ngrams(tokens, 3):
            ctx, nxt = gram[:2], gram[2]
            contexts.setdefault(ctx, []).append(nxt)
        useful = {k: v for k, v in contexts.items() if len(v) >= 2}
        if len(useful) < 4:
            return 0.15
        concentrations: list[float] = []
        for nxts in useful.values():
            counts = Counter(nxts)
            top = max(counts.values())
            concentrations.append(top / len(nxts))
        mean_c = sum(concentrations) / len(concentrations)
        # Random English bigram continuations are diverse; 0.7+ is sticky.
        return clamp((mean_c - 0.35) / 0.50)

    def _window_spikes(self, tokens: list[str], window: int = 48) -> float:
        if len(tokens) < window:
            return 0.0
        seed = DEFAULT_SEEDS[0]
        rates: list[float] = []
        for _, chunk in sliding_windows(tokenize(" ".join(tokens)), window, window // 2):
            words = [t.lower for t in chunk if t.is_word]
            if len(words) < 8:
                continue
            green = sum(
                1 for i in range(1, len(words)) if self._green(words[i - 1], words[i], seed)
            )
            rates.append(green / max(1, len(words) - 1))
        if len(rates) < 2:
            return 0.0
        mean, std = mean_std(rates)
        if std <= 1e-9:
            return 0.0
        peak = max(abs(r - mean) / std for r in rates)
        return clamp((peak - 1.1) / 3.0)

    def _rare_token_alignment(self, tokens: list[str]) -> float:
        """Rare tokens that still land green on many seeds are a hint.

        Tournament watermarks bite hardest on high-entropy / rare tokens.
        If rare tokens are *more* green-aligned than common ones across
        seeds, the passage is a better disruption target.
        """

        from unsynth.data.lexicon import FREQUENCY

        if len(tokens) < 24:
            return 0.0
        rare_hits = 0
        rare_n = 0
        common_hits = 0
        common_n = 0
        seed = DEFAULT_SEEDS[1]
        for i in range(1, len(tokens)):
            tok = tokens[i]
            is_green = self._green(tokens[i - 1], tok, seed)
            rank = FREQUENCY.get(tok)
            if rank is None or rank > 250:
                rare_n += 1
                rare_hits += int(is_green)
            else:
                common_n += 1
                common_hits += int(is_green)
        if rare_n < 6 or common_n < 6:
            return 0.0
        delta = (rare_hits / rare_n) - (common_hits / common_n)
        return clamp(delta / 0.25)

    def _hot_spans(self, text: str, window: int = 80) -> list[SpanSignal]:
        tokens = tokenize(text)
        words = [t for t in tokens if t.is_word]
        if len(words) < window:
            return []
        seed = DEFAULT_SEEDS[0]
        spans: list[SpanSignal] = []
        for start, chunk in sliding_windows(words, window, window // 2):
            w = [t.lower for t in chunk]
            green = sum(1 for i in range(1, len(w)) if self._green(w[i - 1], w[i], seed))
            rate = green / max(1, len(w) - 1)
            z = (rate - GREEN_FRACTION) / math.sqrt(
                GREEN_FRACTION * (1.0 - GREEN_FRACTION) / max(1, len(w) - 1)
            )
            if z >= 1.6:
                spans.append(
                    SpanSignal(
                        start=chunk[0].start,
                        end=chunk[-1].end,
                        score=clamp((z - 1.0) / 4.0),
                        reason=f"local green-rate spike z={z:.2f}",
                    )
                )
            del start
        return spans[:12]


def token_entropy_prior(tokens: list[str]) -> list[float]:
    """Per-token rewrite priority used by the entropy-aware rewriter."""

    from unsynth.data.lexicon import word_entropy_hint

    if not tokens:
        return []
    scores = [word_entropy_hint(tok) for tok in tokens]
    # Boost tokens that sit after a rare context — those are the positions
    # tournament sampling seeds from.
    for i in range(1, len(tokens)):
        if word_entropy_hint(tokens[i - 1]) > 0.6:
            scores[i] = min(1.0, scores[i] + 0.08)
    return scores
