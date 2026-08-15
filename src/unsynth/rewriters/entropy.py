"""Entropy-aware edit targeting.

Generative watermarks (Kirchenbauer green/red lists, SynthID-Text
tournament sampling) inject the most signal where the model already had
a real choice — high local entropy. Changing those positions does two
things at once:

1. Removes a token that was more likely to be hash-favored.
2. Reseeds the context window that every later token is hashed from,
   which cascades. A few well-placed substitutions beat a blanket
   synonym storm.

This module only *ranks* tokens. The lexical / paraphrase layers decide
how to replace them.
"""

from __future__ import annotations

from dataclasses import dataclass

from unsynth.data.lexicon import FREQUENCY, word_entropy_hint
from unsynth.detectors.statistical import token_entropy_prior
from unsynth.text import FUNCTION_WORDS, Token, tokenize


@dataclass(frozen=True, slots=True)
class RankedToken:
    token: Token
    index: int
    priority: float
    reason: str


def rank_tokens(text: str) -> list[RankedToken]:
    tokens = tokenize(text)
    words = [(i, t) for i, t in enumerate(tokens) if t.is_word]
    priors = token_entropy_prior([t.lower for _, t in words])
    ranked: list[RankedToken] = []
    word_i = 0
    for i, tok in enumerate(tokens):
        if not tok.is_word:
            continue
        prior = priors[word_i] if word_i < len(priors) else word_entropy_hint(tok.lower)
        word_i += 1
        reason = "content"
        if tok.lower in FUNCTION_WORDS:
            prior *= 0.15
            reason = "function-word"
        elif FREQUENCY.get(tok.lower, 10_000) <= 80:
            prior *= 0.35
            reason = "closed-class-or-common"
        elif len(tok.text) < 4:
            prior *= 0.25
            reason = "short"
        else:
            reason = "high-entropy-content"
        ranked.append(RankedToken(tok, i, prior, reason))
    ranked.sort(key=lambda r: r.priority, reverse=True)
    return ranked


def select_targets(text: str, strength: float) -> list[RankedToken]:
    """Pick the top fraction of tokens to touch, scaled by strength."""

    ranked = [r for r in rank_tokens(text) if r.priority >= 0.25]
    if not ranked:
        return []
    # strength 0.2 → ~12% of content tokens; 0.9 → ~55%.
    frac = 0.10 + 0.50 * max(0.0, min(1.0, strength))
    k = max(1, int(round(len(ranked) * frac)))
    return ranked[:k]
