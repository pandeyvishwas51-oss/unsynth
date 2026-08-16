"""Detect → rewrite → re-detect loop with adaptive strength."""

from __future__ import annotations

from collections.abc import Callable

from unsynth.config import Settings
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.logging import get_logger
from unsynth.pipeline.document import apply_to_markdown
from unsynth.rewriters.pipeline import RewriteStack
from unsynth.rewriters.quality import QualityGate
from unsynth.safety import sanitize_text
from unsynth.types import (
    DetectionContext,
    DetectorFamily,
    DetectorResult,
    EvalReport,
    PipelineMode,
    PipelineResult,
    RewriteResult,
)

log = get_logger("pipeline")


class UnSynthPipeline:
    """The public orchestrator.

    ``mode=detect``  — score only.
    ``mode=rewrite`` — rewrite once at ``initial_strength`` (still reports scores).
    ``mode=clean``   — adaptive multi-pass until targets are met or ``max_passes``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.ensemble = EnsembleDetector(self.settings)
        self.stack = RewriteStack(self.settings)
        self.quality = QualityGate(self.settings)

    def detect(self, text: str, *, context: DetectionContext | None = None) -> DetectorResult:
        text = sanitize_text(text)
        scored = text
        if self.settings.rewrite.protect_markdown:
            from unsynth.pipeline.markdown import iter_rewriteable, parse_markdown

            prose = "".join(seg.text for _, seg in iter_rewriteable(parse_markdown(text)))
            if prose.strip():
                scored = prose
        return self.ensemble.detect(scored, context=context)

    def rewrite(self, text: str, *, strength: float | None = None) -> RewriteResult:
        text = sanitize_text(text)
        level = self.settings.rewrite.initial_strength if strength is None else strength
        if self.settings.rewrite.protect_markdown:
            holder: list[RewriteResult] = []

            def _one(block: str) -> str:
                result = self.stack.run(block, strength=level)
                holder.append(result)
                return result.rewritten

            rewritten = apply_to_markdown(
                text,
                _one,
                protect_code=self.settings.rewrite.protect_code,
                protect_tables=self.settings.rewrite.protect_tables,
            )
            if holder:
                edits = sum(h.edits for h in holder)
                notes = tuple(n for h in holder for n in h.notes)
                quality = self.quality.evaluate(text, rewritten)
                return RewriteResult(
                    original=text,
                    rewritten=rewritten,
                    strategy=holder[-1].strategy,
                    strength=level,
                    quality=quality,
                    notes=notes,
                    edits=edits,
                )
        return self.stack.run(text, strength=level)

    def run(
        self,
        text: str,
        *,
        mode: PipelineMode = PipelineMode.CLEAN,
        context: DetectionContext | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        def emit(msg: str) -> None:
            log.info("%s", msg)
            if progress:
                progress(msg)

        text = sanitize_text(text)
        before = self.detect(text, context=context)
        emit(f"detect: score={before.score:.3f} label={before.label} conf={before.confidence:.2f}")

        if mode is PipelineMode.DETECT:
            ev = EvalReport(
                before=before,
                after=None,
                quality=None,
                token_stats=None,
                passes=0,
                target_met=False,
                notes=("detect-only",),
            )
            return PipelineResult(
                mode=mode,
                original=text,
                output=text,
                before=before,
                after=None,
                rewrites=(),
                eval=ev,
                target_met=False,
            )

        current = text
        rewrites: list[RewriteResult] = []
        strength = self.settings.rewrite.initial_strength
        after = before
        warnings: list[str] = []

        max_passes = 1 if mode is PipelineMode.REWRITE else self.settings.rewrite.max_passes
        target_met = False
        for pass_i in range(1, max_passes + 1):
            emit(f"pass {pass_i}/{max_passes} strength={strength:.2f}")
            try:
                result = self.rewrite(current, strength=strength)
            except Exception as exc:
                warnings.append(f"pass {pass_i} crashed: {exc}")
                log.warning("rewrite pass %s crashed: %s", pass_i, exc)
                break
            rewrites.append(result)
            if result.rewritten == current and result.edits == 0:
                warnings.append(f"pass {pass_i} made no edits")
            current = result.rewritten
            after = self.detect(current, context=context)
            emit(
                f"re-detect: score={after.score:.3f} label={after.label} "
                f"sim={result.quality.similarity:.3f} change={result.quality.token_change_rate:.3f}"
            )
            if _targets_met(self.settings, after, result.quality.passed):
                target_met = True
                emit("targets met")
                break
            if not result.quality.passed:
                warnings.append(f"pass {pass_i} quality: {', '.join(result.quality.reasons)}")
                # Quality failed after markdown reassembly — do not keep climbing
                # blindly; take a smaller step.
                strength = min(
                    self.settings.rewrite.max_strength,
                    strength + self.settings.rewrite.strength_step * 0.5,
                )
            else:
                strength = min(
                    self.settings.rewrite.max_strength,
                    strength + self.settings.rewrite.strength_step,
                )
                # If watermark score is the stubborn one, bias later passes
                # toward lexical + structural (already in the stack).
                if _watermark_score(after) > self.settings.rewrite.target_watermark_score:
                    emit("watermark heuristic still high; increasing structural pressure")

        quality = self.quality.evaluate(text, current)
        stats = self.quality.token_stats(text, current)
        ev = EvalReport(
            before=before,
            after=after,
            quality=quality,
            token_stats=stats,
            passes=len(rewrites),
            target_met=target_met,
            notes=tuple(warnings),
        )
        if not target_met:
            warnings.append("targets not fully met; this is a disruption tool, not a guarantee")
        return PipelineResult(
            mode=mode,
            original=text,
            output=current,
            before=before,
            after=after,
            rewrites=tuple(rewrites),
            eval=ev,
            target_met=target_met,
            warnings=tuple(warnings),
        )


def _watermark_score(result: DetectorResult) -> float:
    members = result.details.get("members") if isinstance(result.details, dict) else None
    if not isinstance(members, list):
        return result.score if result.family is DetectorFamily.WATERMARK else 0.0
    vals = [
        float(m["score"])
        for m in members
        if isinstance(m, dict) and m.get("family") == DetectorFamily.WATERMARK.value
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _classical_score(result: DetectorResult) -> float:
    members = result.details.get("members") if isinstance(result.details, dict) else None
    if not isinstance(members, list):
        return result.score
    vals = [
        float(m["score"])
        for m in members
        if isinstance(m, dict)
        and m.get("family") in {DetectorFamily.CLASSICAL.value, DetectorFamily.STYLOMETRIC.value}
    ]
    return sum(vals) / len(vals) if vals else result.score


def _targets_met(settings: Settings, after: DetectorResult, quality_ok: bool) -> bool:
    if not quality_ok:
        return False
    classic = _classical_score(after)
    mark = _watermark_score(after)
    return (
        classic <= settings.rewrite.target_ai_score
        and mark <= settings.rewrite.target_watermark_score
    )


def run_pipeline(
    text: str,
    *,
    settings: Settings | None = None,
    mode: PipelineMode = PipelineMode.CLEAN,
) -> PipelineResult:
    return UnSynthPipeline(settings).run(text, mode=mode)
