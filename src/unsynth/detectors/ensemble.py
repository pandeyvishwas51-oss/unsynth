"""Weighted ensemble over the configured detectors."""

from __future__ import annotations

from collections.abc import Sequence

from unsynth.config import Settings
from unsynth.detectors.base import BaseDetector
from unsynth.detectors.registry import DetectorRegistry
from unsynth.logging import get_logger
from unsynth.types import (
    DetectionContext,
    DetectorFamily,
    DetectorResult,
    Signal,
)

log = get_logger("ensemble")

FAMILY_WEIGHTS: dict[DetectorFamily, float] = {
    DetectorFamily.CLASSICAL: 0.34,
    DetectorFamily.STYLOMETRIC: 0.28,
    DetectorFamily.WATERMARK: 0.30,
    DetectorFamily.API: 0.50,  # if a vendor API is actually configured
    DetectorFamily.ENSEMBLE: 0.0,
}


class EnsembleDetector(BaseDetector):
    name = "ensemble"
    family = DetectorFamily.ENSEMBLE
    version = "1.0.0"

    def __init__(
        self,
        settings: Settings | None = None,
        detectors: Sequence[BaseDetector] | None = None,
    ) -> None:
        super().__init__(settings)
        if detectors is not None:
            self.detectors = list(detectors)
        else:
            registry = DetectorRegistry(self.settings)
            names = [n for n in self.settings.detect.detectors if n != "ensemble"]
            self.detectors = registry.create_many(names)

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        children: list[DetectorResult] = []
        for det in self.detectors:
            try:
                children.append(det.detect(text, context=context))
            except Exception as exc:
                log.warning("detector %s crashed: %s", det.name, exc)
                children.append(BaseDetector.empty_result(det.name, det.family, f"crashed: {exc}"))
        if not children:
            return self.empty_result(self.name, self.family, "no detectors configured")

        usable = [c for c in children if c.confidence > 0.0 or c.score > 0.0]
        if not usable:
            usable = children

        weighted_score = 0.0
        weighted_conf = 0.0
        weight_sum = 0.0
        family_scores: dict[str, list[float]] = {}
        for child in usable:
            # Idle API adapters (confidence 0, score 0, unconfigured) do not vote.
            if child.family is DetectorFamily.API and child.confidence <= 0.0:
                continue
            weight = FAMILY_WEIGHTS.get(child.family, 0.2) * max(0.15, child.confidence)
            weighted_score += child.score * weight
            weighted_conf += child.confidence * weight
            weight_sum += weight
            family_scores.setdefault(child.family.value, []).append(child.score)

        if weight_sum <= 0:
            score = sum(c.score for c in children) / len(children)
            confidence = 0.2
        else:
            score = weighted_score / weight_sum
            confidence = weighted_conf / weight_sum

        # Agreement bonus: detectors pointing the same way raise confidence.
        scores = [c.score for c in usable]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
        agreement = max(0.0, 1.0 - (var**0.5) / 0.35)
        confidence = max(0.0, min(1.0, 0.7 * confidence + 0.3 * agreement))

        family_means = {
            name: round(sum(vals) / len(vals), 4) for name, vals in family_scores.items()
        }
        signals = tuple(
            Signal(c.name, c.score, FAMILY_WEIGHTS.get(c.family, 0.2), c.label) for c in children
        )
        # Promote the strongest child spans.
        spans = tuple(sorted((s for c in children for s in c.spans), key=lambda s: -s.score)[:16])

        watermark_mean = family_means.get("watermark", 0.0)
        force_watermark = (
            watermark_mean >= self.settings.detect.watermark_likely and watermark_mean >= score
        )
        return self.result(
            score,
            confidence=confidence,
            details={
                "members": [c.as_dict() for c in children],
                "family_means": family_means,
                "agreement": round(agreement, 4),
            },
            signals=signals,
            spans=spans,
            family=DetectorFamily.WATERMARK if force_watermark else None,
        )

    def family_score(self, result: DetectorResult, family: DetectorFamily) -> float:
        members = result.details.get("members") if isinstance(result.details, dict) else None
        if not isinstance(members, list):
            return result.score
        vals = [
            float(m["score"])
            for m in members
            if isinstance(m, dict) and m.get("family") == family.value
        ]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)
