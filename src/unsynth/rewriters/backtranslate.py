"""Optional pivot-language paraphrase. Off unless explicitly enabled.

Back-translation is a classic detector-evasion trick and also a good way
to wreck meaning. It is gated by ``rewrite.allow_backtranslate`` and
still requires a local backend.
"""

from __future__ import annotations

from unsynth.backends import complete, is_available
from unsynth.rewriters.base import BaseRewriter
from unsynth.types import RewriteResult

TO_PIVOT = (
    "Translate the passage into idiomatic French. Keep names and numbers. "
    "Return only the translation.\n\n{text}"
)
FROM_PIVOT = (
    "Translate the passage into natural, slightly informal English. "
    "Keep names and numbers. Return only the translation.\n\n{text}"
)


class BacktranslateRewriter(BaseRewriter):
    name = "backtranslate"
    requires_backend = True

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        if not self.settings.rewrite.allow_backtranslate:
            return self.pack(text, text, strength=strength, notes=("disabled",))
        if not is_available(self.settings):
            return self.pack(text, text, strength=strength, notes=("backend-unavailable",))
        try:
            french = complete(self.settings, TO_PIVOT.format(text=text.strip()), temperature=0.4)
            english = complete(
                self.settings,
                FROM_PIVOT.format(text=french.strip()),
                temperature=0.7 + 0.2 * strength,
            )
        except Exception as exc:
            return self.pack(text, text, strength=strength, notes=(f"backend-error:{exc}",))
        out = english.strip()
        if not out:
            return self.pack(text, text, strength=strength, notes=("empty-backend",))
        return self.pack(text, out, strength=strength, edits=2, notes=("fr-en",))
