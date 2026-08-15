"""Shared dataclasses and enums used across UnSynth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

Label = Literal["human", "ai", "watermarked", "uncertain"]
SegmentKind = Literal[
    "prose",
    "code",
    "table",
    "heading",
    "list_item",
    "blockquote",
    "html",
    "frontmatter",
    "url",
    "other",
]


class DetectorFamily(str, Enum):
    """High-level detector family.

    Classical detectors (perplexity/burstiness/ZeroGPT-style) are a
    completely different object from generative watermarks. Mixing the
    two scores without labeling the family is how most tools mislead users.
    """

    CLASSICAL = "classical"
    STYLOMETRIC = "stylometric"
    WATERMARK = "watermark"
    API = "api"
    ENSEMBLE = "ensemble"


class BackendKind(str, Enum):
    NONE = "none"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    TRANSFORMERS = "transformers"


class PipelineMode(str, Enum):
    DETECT = "detect"
    REWRITE = "rewrite"
    CLEAN = "clean"


@dataclass(frozen=True, slots=True)
class Signal:
    """A named, inspectable feature contributing to a detector score."""

    name: str
    value: float
    weight: float = 1.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class SpanSignal:
    """A localized region of text that looks suspicious."""

    start: int
    end: int
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Output of a single detector (or the ensemble)."""

    name: str
    family: DetectorFamily
    score: float
    confidence: float
    label: Label
    details: Mapping[str, Any] = field(default_factory=dict)
    signals: tuple[Signal, ...] = ()
    spans: tuple[SpanSignal, ...] = ()
    version: str = "1.0.0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "details": dict(self.details),
            "signals": [
                {
                    "name": s.name,
                    "value": round(s.value, 4),
                    "weight": s.weight,
                    "note": s.note,
                }
                for s in self.signals
            ],
            "spans": [
                {
                    "start": s.start,
                    "end": s.end,
                    "score": round(s.score, 4),
                    "reason": s.reason,
                }
                for s in self.spans
            ],
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Semantic / readability / length checks on a rewrite candidate."""

    similarity: float
    readability_before: float
    readability_after: float
    length_ratio: float
    token_change_rate: float
    passed: bool
    reasons: tuple[str, ...] = ()
    method: str = "tfidf"

    def as_dict(self) -> dict[str, Any]:
        return {
            "similarity": round(self.similarity, 4),
            "readability_before": round(self.readability_before, 2),
            "readability_after": round(self.readability_after, 2),
            "length_ratio": round(self.length_ratio, 4),
            "token_change_rate": round(self.token_change_rate, 4),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Output of one rewrite strategy or a full rewrite pass."""

    original: str
    rewritten: str
    strategy: str
    strength: float
    quality: QualityReport
    notes: tuple[str, ...] = ()
    edits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strength": round(self.strength, 3),
            "edits": self.edits,
            "quality": self.quality.as_dict(),
            "notes": list(self.notes),
            "original_chars": len(self.original),
            "rewritten_chars": len(self.rewritten),
        }


@dataclass(frozen=True, slots=True)
class TokenChangeStats:
    original_tokens: int
    rewritten_tokens: int
    changed_tokens: int
    change_rate: float
    unigram_jaccard: float
    char_ngram_cosine: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "rewritten_tokens": self.rewritten_tokens,
            "changed_tokens": self.changed_tokens,
            "change_rate": round(self.change_rate, 4),
            "unigram_jaccard": round(self.unigram_jaccard, 4),
            "char_ngram_cosine": round(self.char_ngram_cosine, 4),
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Before/after evaluation for a detect+rewrite run."""

    before: DetectorResult
    after: DetectorResult | None
    quality: QualityReport | None
    token_stats: TokenChangeStats | None
    passes: int
    target_met: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.as_dict(),
            "after": self.after.as_dict() if self.after else None,
            "quality": self.quality.as_dict() if self.quality else None,
            "token_stats": self.token_stats.as_dict() if self.token_stats else None,
            "passes": self.passes,
            "target_met": self.target_met,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Full orchestrator output."""

    mode: PipelineMode
    original: str
    output: str
    before: DetectorResult
    after: DetectorResult | None
    rewrites: tuple[RewriteResult, ...]
    eval: EvalReport
    target_met: bool
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "target_met": self.target_met,
            "warnings": list(self.warnings),
            "before": self.before.as_dict(),
            "after": self.after.as_dict() if self.after else None,
            "rewrites": [r.as_dict() for r in self.rewrites],
            "eval": self.eval.as_dict(),
            "original_chars": len(self.original),
            "output_chars": len(self.output),
        }


@dataclass(frozen=True, slots=True)
class Segment:
    """A document slice that may be protected from rewriting."""

    kind: SegmentKind
    text: str
    protected: bool
    start: int = 0
    end: int = 0


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """Optional extra information passed to detectors."""

    source_path: str | None = None
    language: str = "en"
    window_tokens: int = 256
    extra: Mapping[str, Any] = field(default_factory=dict)
