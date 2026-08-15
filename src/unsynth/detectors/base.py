"""Detector protocol and shared scoring helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from unsynth.config import Settings
from unsynth.text import Token, sliding_windows, tokenize
from unsynth.types import (
    DetectionContext,
    DetectorFamily,
    DetectorResult,
    Label,
    Signal,
    SpanSignal,
)


class BaseDetector(ABC):
    """All detectors implement this surface.

    New families (OpenAI, Google SynthID official API, vendor classifiers)
    should subclass this rather than reaching into the pipeline.
    """

    name: str = "base"
    family: DetectorFamily = DetectorFamily.CLASSICAL
    version: str = "1.0.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @abstractmethod
    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        """Score a full document."""

    def detect_windows(
        self,
        text: str,
        *,
        context: DetectionContext | None = None,
        window_tokens: int | None = None,
        stride: int | None = None,
    ) -> list[DetectorResult]:
        ctx = context or DetectionContext()
        window = window_tokens or ctx.window_tokens or self.settings.detect.window_tokens
        step = stride or self.settings.detect.window_stride
        tokens = tokenize(text)
        results: list[DetectorResult] = []
        for _, chunk in sliding_windows(tokens, window, step):
            if not chunk:
                continue
            piece = text[chunk[0].start : chunk[-1].end]
            results.append(self.detect(piece, context=ctx))
        return results or [self.detect(text, context=ctx)]

    def label(self, score: float) -> Label:
        watermark = self.family is DetectorFamily.WATERMARK
        value = self.settings.label_for(score, family_watermark=watermark)
        return value  # type: ignore[return-value]

    def result(
        self,
        score: float,
        *,
        confidence: float,
        details: dict[str, object] | None = None,
        signals: Sequence[Signal] = (),
        spans: Sequence[SpanSignal] = (),
        family: DetectorFamily | None = None,
        label: Label | None = None,
    ) -> DetectorResult:
        used_family = family or self.family
        used_label = label or self.label(score)
        if family is DetectorFamily.WATERMARK and label is None:
            used_label = self.settings.label_for(score, family_watermark=True)  # type: ignore[assignment]
        return DetectorResult(
            name=self.name,
            family=used_family,
            score=float(max(0.0, min(1.0, score))),
            confidence=float(max(0.0, min(1.0, confidence))),
            label=used_label,
            details=details or {},
            signals=tuple(signals),
            spans=tuple(spans),
            version=self.version,
        )

    @staticmethod
    def empty_result(name: str, family: DetectorFamily, reason: str) -> DetectorResult:
        return DetectorResult(
            name=name,
            family=family,
            score=0.0,
            confidence=0.0,
            label="uncertain",
            details={"reason": reason},
        )

    def window_spans(
        self, text: str, tokens: Sequence[Token], scores: Sequence[float]
    ) -> list[SpanSignal]:
        spans: list[SpanSignal] = []
        if not tokens or not scores:
            return spans
        # Best-effort: treat a single document score as one span.
        if len(scores) == 1:
            return [SpanSignal(0, len(text), scores[0], f"{self.name} document score")]
        return spans
