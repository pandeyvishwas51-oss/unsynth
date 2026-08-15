"""Rewriter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from unsynth.config import Settings
from unsynth.rewriters.quality import QualityGate
from unsynth.types import QualityReport, RewriteResult


class BaseRewriter(ABC):
    name: str = "base"
    requires_backend: bool = False
    version: str = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.quality = QualityGate(self.settings)

    @abstractmethod
    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        """Return a rewritten candidate. Must not raise on ordinary prose."""

    def pack(
        self,
        original: str,
        rewritten: str,
        *,
        strength: float,
        edits: int = 0,
        notes: tuple[str, ...] = (),
        quality: QualityReport | None = None,
    ) -> RewriteResult:
        report = quality or self.quality.evaluate(original, rewritten)
        return RewriteResult(
            original=original,
            rewritten=rewritten,
            strategy=self.name,
            strength=strength,
            quality=report,
            notes=notes,
            edits=edits,
        )
