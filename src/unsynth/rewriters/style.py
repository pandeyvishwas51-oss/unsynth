"""Style-variance injector — defeats stylometric / burstiness detectors."""

from __future__ import annotations

import re

from unsynth.data.lexicon import CONTRACTIONS, HUMAN_ASIDES, STRUCTURE_TRANSITIONS
from unsynth.rewriters.base import BaseRewriter
from unsynth.text import seeded_rng, sentences, words
from unsynth.types import RewriteResult

_FIRST_PERSON = re.compile(r"\b(I|we|our|my)\b")


class StyleHumanizer(BaseRewriter):
    name = "style"
    requires_backend = False

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        rng = seeded_rng(text[:80], "style", round(strength * 100), seed=self.settings.runtime.seed)
        sents = sentences(text)
        if not sents:
            return self.pack(text, text, strength=strength, edits=0)

        out: list[str] = []
        edits = 0
        used_aside = False
        used_transition = False
        for i, sent in enumerate(sents):
            working = sent
            rolled = rng.random()
            # Contractions: the single highest-leverage stylometric tell.
            contracted, n = self._contract(working, rng, strength)
            if n:
                working = contracted
                edits += n
            # At most one aside per block — more than that looks templated.
            if (
                i > 0
                and not used_aside
                and rolled < 0.06 + 0.10 * strength
                and len(words(working)) > 12
            ):
                aside = rng.choice(HUMAN_ASIDES)
                working = self._insert_aside(working, aside)
                used_aside = True
                edits += 1
            # At most one opener tweak, and never a fake extra sentence.
            first = words(working)[0].lower() if words(working) else ""
            busy_openers = {
                "and",
                "but",
                "so",
                "also",
                "plus",
                "then",
                "still",
                "worth",
                "wrapping",
                "note",
            }
            if (
                i == 1
                and not used_transition
                and first not in busy_openers
                and rng.random() < 0.12 + 0.18 * strength
            ):
                trans = rng.choice(STRUCTURE_TRANSITIONS).rstrip(".,:")
                if not working.lower().startswith(trans.lower()[:6]):
                    rest = working[0].lower() + working[1:] if working[:1].isupper() else working
                    working = f"{trans[0].upper()}{trans[1:]}, {rest}"
                    used_transition = True
                    edits += 1
            if ";" in working and rng.random() < 0.4 * strength:
                working = working.replace(";", " —", 1)
                edits += 1
            elif len(words(working)) > 22 and rng.random() < 0.15 * strength:
                working = self._break_with_dash(working)
                edits += 1
            out.append(working)

        rewritten = " ".join(out)
        return self.pack(text, rewritten, strength=strength, edits=edits)

    def _contract(self, sent: str, rng: object, strength: float) -> tuple[str, int]:
        edits = 0
        out = sent
        # Prefer contracting; expanding is only for variety at high strength.
        items = sorted(CONTRACTIONS.items(), key=lambda kv: len(kv[0]), reverse=True)
        for full, short in items:
            pattern = re.compile(rf"\b{re.escape(full)}\b", re.IGNORECASE)
            if not pattern.search(out):
                continue
            if rng.random() > 0.35 + 0.5 * strength:  # type: ignore[attr-defined]
                continue

            def _sub(match: re.Match[str], short: str = short) -> str:
                src = match.group(0)
                if src[0].isupper():
                    return short[0].upper() + short[1:]
                return short

            out, n = pattern.subn(_sub, out, count=1)
            edits += n
        return out, edits

    def _insert_aside(self, sent: str, aside: str) -> str:
        # Slide the aside after the first clause if there's a comma.
        comma = sent.find(",")
        if 12 <= comma <= 80:
            return f"{sent[:comma]} ({aside}){sent[comma:]}"
        stripped = sent.rstrip(".!?")
        end = sent[len(stripped) :] or "."
        return f"{stripped} — {aside}{end}"

    def _break_with_dash(self, sent: str) -> str:
        comma = sent.find(",")
        if comma > 10:
            return sent[:comma] + " —" + sent[comma + 1 :]
        return sent
