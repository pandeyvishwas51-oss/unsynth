"""Sentence split / merge / reorder — destroys n-gram context windows."""

from __future__ import annotations

import re

from unsynth.rewriters.base import BaseRewriter
from unsynth.text import safe_join_sentences, seeded_rng, sentences, words
from unsynth.types import RewriteResult

_COORD = re.compile(
    r"\s+,\s+(and|but|so|yet)\s+",
    re.IGNORECASE,
)
_SUBORD = re.compile(
    r"^(Although|While|Because|When|If|Though)\s+(.+?),\s+(.+)$",
    re.IGNORECASE,
)


class StructuralRewriter(BaseRewriter):
    name = "structural"
    requires_backend = False

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        rng = seeded_rng(
            text[:80], "structural", round(strength * 100), seed=self.settings.runtime.seed
        )
        sents = sentences(text)
        if len(sents) < 2:
            return self.pack(text, text, strength=strength, edits=0, notes=("too-few-sentences",))

        out: list[str] = []
        edits = 0
        i = 0
        while i < len(sents):
            sent = sents[i]
            roll = rng.random()
            # Split long coordinated sentences.
            if roll < 0.25 + 0.35 * strength and len(words(sent)) >= 18:
                split = self._try_split(sent)
                if split:
                    out.extend(split)
                    edits += 1
                    i += 1
                    continue
            # Merge a short sentence into the next one.
            if (
                i + 1 < len(sents)
                and roll < 0.18 + 0.25 * strength
                and len(words(sent)) <= 10
                and len(words(sents[i + 1])) <= 22
            ):
                merged = self._merge(sent, sents[i + 1], rng)
                out.append(merged)
                edits += 1
                i += 2
                continue
            # Flip "Although X, Y" → "Y, although X".
            if roll < 0.20 + 0.30 * strength:
                flipped = self._flip_subordinate(sent)
                if flipped and flipped != sent:
                    out.append(flipped)
                    edits += 1
                    i += 1
                    continue
            out.append(sent)
            i += 1

        # Light adjacent swap of independent short sentences at high strength.
        if strength >= 0.7 and len(out) >= 4:
            for j in range(0, len(out) - 1, 3):
                if len(words(out[j])) <= 14 and len(words(out[j + 1])) <= 14:
                    if rng.random() < 0.35:
                        out[j], out[j + 1] = out[j + 1], out[j]
                        edits += 1

        rewritten = safe_join_sentences(out)
        # Preserve trailing newline style of the original paragraph-ish input.
        if text.endswith("\n") and not rewritten.endswith("\n"):
            rewritten += "\n"
        return self.pack(text, rewritten, strength=strength, edits=edits)

    def _try_split(self, sent: str) -> list[str] | None:
        match = _COORD.search(sent)
        if not match:
            return None
        left = sent[: match.start()].strip()
        right = sent[match.end() :].strip()
        conj = match.group(1).lower()
        if len(words(left)) < 6 or len(words(right)) < 5:
            return None
        if not left.endswith((".", "!", "?")):
            left = left.rstrip(",;") + "."
        if conj == "but" or conj == "so":
            right = right[0].upper() + right[1:]
        else:
            right = right[0].upper() + right[1:]
        if not right.endswith((".", "!", "?")):
            right += "."
        return [left, right]

    def _merge(self, a: str, b: str, rng: object) -> str:
        left = a.rstrip(" .!?")
        right = b[0].lower() + b[1:] if b[:1].isupper() and (len(b) < 2 or b[1:2].islower()) else b
        glue = str(rng.choice([", and ", "; ", " — ", ", so "]))  # type: ignore[attr-defined]
        merged = left + glue + right
        if not merged.endswith((".", "!", "?")):
            merged += "."
        return merged

    def _flip_subordinate(self, sent: str) -> str | None:
        match = _SUBORD.match(sent.strip())
        if not match:
            return None
        sub, dep, main = match.group(1), match.group(2), match.group(3)
        main = main.rstrip(".")
        dep = dep[0].lower() + dep[1:] if dep else dep
        return f"{main}, {sub.lower()} {dep}."
