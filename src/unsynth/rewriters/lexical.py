"""Entropy-aware synonym / phrase layer. Works with no language model."""

from __future__ import annotations

import re

from unsynth.data.lexicon import AI_PHRASES, SYNONYMS, TRANSITION_SWAPS
from unsynth.rewriters.base import BaseRewriter
from unsynth.rewriters.entropy import select_targets
from unsynth.text import match_casing, reconstruct, seeded_rng
from unsynth.types import RewriteResult

_WORD_BOUNDARY = re.compile(r"\b", re.UNICODE)


class LexicalRewriter(BaseRewriter):
    name = "lexical"
    requires_backend = False

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        rng = seeded_rng(
            text[:80], "lexical", round(strength * 100), seed=self.settings.runtime.seed
        )
        working, phrase_edits = self._replace_phrases(text, strength, rng)
        working, word_edits = self._replace_words(working, strength, rng)
        notes = []
        if phrase_edits:
            notes.append(f"phrases={phrase_edits}")
        if word_edits:
            notes.append(f"words={word_edits}")
        return self.pack(
            text,
            working,
            strength=strength,
            edits=phrase_edits + word_edits,
            notes=tuple(notes),
        )

    def _replace_phrases(self, text: str, strength: float, rng: object) -> tuple[str, int]:
        # Longer phrases first so we don't nibble them into words.
        keys = sorted(AI_PHRASES.keys(), key=len, reverse=True)
        edits = 0
        out = text
        # Strength gates how aggressively we rewrite stock phrases.
        # Always rewrite the most toxic ones; sample the rest.
        for phrase in keys:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = list(pattern.finditer(out))
            if not matches:
                continue
            for match in reversed(matches):
                # Stock phrases are the highest-leverage classical-detector
                # tell. Always rewrite them; strength only picks the variant.
                options = AI_PHRASES[phrase]
                repl = match_casing(match.group(0), rng.choice(options))  # type: ignore[attr-defined]
                out = out[: match.start()] + repl + out[match.end() :]
                edits += 1
        return out, edits

    def _replace_words(self, text: str, strength: float, rng: object) -> tuple[str, int]:
        targets = select_targets(text, strength)
        if not targets:
            return text, 0
        replacements: list[tuple[int, int, str]] = []
        used_spans: list[tuple[int, int]] = []
        for ranked in targets:
            tok = ranked.token
            options = SYNONYMS.get(tok.lower) or TRANSITION_SWAPS.get(tok.lower)
            if not options:
                continue
            if any(not (tok.end <= s or tok.start >= e) for s, e in used_spans):
                continue
            # Keep some originals so the passage doesn't become a thesaurus.
            keep_p = 0.55 - 0.35 * strength
            if rng.random() < keep_p:  # type: ignore[attr-defined]
                continue
            repl = match_casing(tok.text, rng.choice(options))  # type: ignore[attr-defined]
            if repl.lower() == tok.lower:
                continue
            replacements.append((tok.start, tok.end, repl))
            used_spans.append((tok.start, tok.end))
        return reconstruct(text, replacements), len(replacements)
