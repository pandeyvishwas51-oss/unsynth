"""Before/after reports for researchers and the CLI."""

from __future__ import annotations

from unsynth.config import Settings
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.rewriters.quality import QualityGate
from unsynth.types import DetectorResult, EvalReport, PipelineResult


def compare_texts(
    original: str,
    rewritten: str,
    *,
    settings: Settings | None = None,
) -> EvalReport:
    cfg = settings or Settings()
    ensemble = EnsembleDetector(cfg)
    quality = QualityGate(cfg)
    before = ensemble.detect(original)
    after = ensemble.detect(rewritten)
    q = quality.evaluate(original, rewritten)
    stats = quality.token_stats(original, rewritten)
    return EvalReport(
        before=before,
        after=after,
        quality=q,
        token_stats=stats,
        passes=1,
        target_met=q.passed and after.score <= cfg.rewrite.target_ai_score,
    )


def render_markdown_report(result: PipelineResult) -> str:
    before = result.before
    after = result.after
    lines = [
        "# UnSynth report",
        "",
        f"- Mode: `{result.mode.value}`",
        f"- Target met: **{result.target_met}**",
        f"- Passes: {result.eval.passes}",
        "",
        "## Scores",
        "",
        "| | score | label | confidence |",
        "|---|---:|---|---:|",
        f"| before | {before.score:.3f} | {before.label} | {before.confidence:.2f} |",
    ]
    if after is not None:
        lines.append(f"| after | {after.score:.3f} | {after.label} | {after.confidence:.2f} |")
    lines.extend(["", "### Detector members (before)", ""])
    lines.extend(_member_table(before))
    if after is not None:
        lines.extend(["", "### Detector members (after)", ""])
        lines.extend(_member_table(after))
    if result.eval.quality is not None:
        q = result.eval.quality
        lines.extend(
            [
                "",
                "## Quality",
                "",
                f"- Similarity ({q.method}): **{q.similarity:.3f}**",
                f"- Token change rate: {q.token_change_rate:.3f}",
                f"- Length ratio: {q.length_ratio:.3f}",
                f"- Readability: {q.readability_before:.1f} → {q.readability_after:.1f}",
                f"- Gate passed: {q.passed}",
            ]
        )
        if q.reasons:
            lines.append("- Reasons: " + "; ".join(q.reasons))
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in result.warnings)
    lines.extend(
        [
            "",
            "## Honesty note",
            "",
            "UnSynth disrupts *signals*. It does not cryptographically erase a",
            "private-key generative watermark. A vendor with the secret key may",
            "still score residual alignment, especially on lightly edited text.",
            "",
        ]
    )
    return "\n".join(lines)


def _member_table(result: DetectorResult) -> list[str]:
    members = result.details.get("members") if isinstance(result.details, dict) else None
    lines = [
        "| detector | family | score | label | confidence |",
        "|---|---|---:|---|---:|",
    ]
    if isinstance(members, list):
        for m in members:
            if not isinstance(m, dict):
                continue
            lines.append(
                f"| {m.get('name')} | {m.get('family')} | {float(m.get('score', 0)):.3f} "
                f"| {m.get('label')} | {float(m.get('confidence', 0)):.2f} |"
            )
    else:
        lines.append(
            f"| {result.name} | {result.family.value} | {result.score:.3f} "
            f"| {result.label} | {result.confidence:.2f} |"
        )
    return lines
