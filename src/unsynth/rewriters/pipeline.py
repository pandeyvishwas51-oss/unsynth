"""Compose rewrite strategies into a single pass at a given strength."""

from __future__ import annotations

from collections.abc import Sequence

from unsynth.config import Settings
from unsynth.rewriters.base import BaseRewriter
from unsynth.rewriters.quality import QualityGate
from unsynth.rewriters.registry import RewriterRegistry
from unsynth.types import RewriteResult


class RewriteStack:
    """Run configured strategies in order, keeping the last quality-passing text.

    Strategies that require a backend and don't have one are skipped.
    If a strategy fails the quality gate we keep the previous text.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        strategies: Sequence[BaseRewriter] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.quality = QualityGate(self.settings)
        if strategies is not None:
            self.strategies = list(strategies)
        else:
            registry = RewriterRegistry(self.settings)
            names = list(self.settings.rewrite.strategies)
            if self.settings.rewrite.allow_backtranslate and "backtranslate" not in names:
                names.append("backtranslate")
            self.strategies = registry.create_many(names)

    def run(self, text: str, *, strength: float) -> RewriteResult:
        current = text
        total_edits = 0
        notes: list[str] = []
        last_quality = self.quality.evaluate(text, text)
        applied: list[str] = []
        for strategy in self.strategies:
            if strategy.requires_backend:
                from unsynth.backends import is_available

                if not is_available(self.settings):
                    notes.append(f"{strategy.name}:skipped-no-backend")
                    continue
            result = strategy.rewrite(current, strength=strength)
            if result.rewritten == current:
                notes.append(f"{strategy.name}:noop")
                continue
            gate = self.quality.evaluate(text, result.rewritten)
            # Lexical/style exist to kill high-IDF stock phrases. A TF-IDF
            # gate at 0.82 would reject the exact edits we want. Keep hard
            # fails (empty, length collapse) and use a softer floor.
            soft = strategy.name in {"lexical", "style"}
            floor = self.settings.rewrite.min_similarity * (0.70 if soft else 1.0)
            hard = [r for r in gate.reasons if not r.startswith("similarity ")]
            if hard or gate.similarity < floor:
                why = ",".join(hard) if hard else f"similarity {gate.similarity:.3f}"
                notes.append(f"{strategy.name}:rejected({why})")
                continue
            current = result.rewritten
            total_edits += result.edits
            last_quality = gate
            applied.append(strategy.name)
            notes.extend(f"{strategy.name}:{n}" for n in result.notes)
        return RewriteResult(
            original=text,
            rewritten=current,
            strategy="+".join(applied) or "identity",
            strength=strength,
            quality=last_quality,
            notes=tuple(notes),
            edits=total_edits,
        )
