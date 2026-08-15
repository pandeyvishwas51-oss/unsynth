"""Hard production invariants. If these fail, we do not ship."""

from __future__ import annotations

from tests.helpers import (
    AI_PARAGRAPH,
    HUMAN_PARAGRAPH,
    MARKDOWN_DOC,
    assert_pipeline_sane,
    assert_protected_intact,
    assert_result_sane,
    mixed_unicode_doc,
    repeat_article,
)
from unsynth.config import Settings
from unsynth.detectors.classical import ClassicalDetector
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.detectors.statistical import StatisticalWatermarkDetector
from unsynth.detectors.stylometric import StylometricDetector
from unsynth.pipeline.markdown import parse_markdown, render_segments
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.text import seeded_rng, tfidf_cosine
from unsynth.types import PipelineMode


def _pipe(settings: Settings) -> UnSynthPipeline:
    return UnSynthPipeline(settings)


def test_markdown_parse_is_lossless() -> None:
    segs = parse_markdown(MARKDOWN_DOC)
    assert render_segments(segs) == MARKDOWN_DOC
    kinds = {s.kind for s in segs}
    assert "code" in kinds
    assert "table" in kinds
    assert "frontmatter" in kinds
    assert "heading" in kinds


def test_clean_preserves_protected_regions(settings: Settings) -> None:
    result = _pipe(settings).run(MARKDOWN_DOC, mode=PipelineMode.CLEAN)
    assert_pipeline_sane(result)
    assert_protected_intact(
        MARKDOWN_DOC,
        result.output,
        [
            'SECRET_TOKEN = "do-not-touch"',
            "| metric | value |",
            "| kept | yes |",
            "# Heading Must Survive",
            "title: keep-me",
            "https://example.com/path?q=1",
            "ops@example.com",
        ],
    )


def test_detectors_never_emit_illegal_scores(settings: Settings) -> None:
    texts = [
        "",
        "   ",
        "short",
        AI_PARAGRAPH,
        HUMAN_PARAGRAPH,
        mixed_unicode_doc(),
        "\x00\x00" + AI_PARAGRAPH,
        "a " * 20,
        "🎉" * 40,
        "the the the the the the the the the the the the",
    ]
    dets = [
        ClassicalDetector(settings),
        StylometricDetector(settings),
        StatisticalWatermarkDetector(settings),
        EnsembleDetector(settings),
    ]
    for text in texts:
        for det in dets:
            assert_result_sane(det.detect(text))


def test_rewrite_is_deterministic_for_fixed_seed(settings: Settings) -> None:
    pipe = _pipe(settings)
    a = pipe.rewrite(AI_PARAGRAPH, strength=0.7)
    b = pipe.rewrite(AI_PARAGRAPH, strength=0.7)
    assert a.rewritten == b.rewritten
    assert a.edits == b.edits


def test_seeded_rng_stable() -> None:
    r1 = seeded_rng("abc", 1, seed=7)
    r2 = seeded_rng("abc", 1, seed=7)
    assert [r1.random() for _ in range(8)] == [r2.random() for _ in range(8)]


def test_pipeline_honors_max_passes(settings: Settings) -> None:
    settings.rewrite.max_passes = 2
    settings.rewrite.target_ai_score = 0.0
    settings.rewrite.target_watermark_score = 0.0
    result = _pipe(settings).run(AI_PARAGRAPH, mode=PipelineMode.CLEAN)
    assert result.eval.passes <= 2


def test_detect_only_does_not_mutate(settings: Settings) -> None:
    src = MARKDOWN_DOC
    result = _pipe(settings).run(src, mode=PipelineMode.DETECT)
    assert result.output == src
    assert result.rewrites == ()


def test_json_roundtrip_pipeline(settings: Settings) -> None:
    result = _pipe(settings).run(AI_PARAGRAPH, mode=PipelineMode.REWRITE)
    assert_pipeline_sane(result)


def test_classical_separates_ai_and_human(settings: Settings) -> None:
    det = ClassicalDetector(settings)
    ai = det.detect(AI_PARAGRAPH)
    human = det.detect(HUMAN_PARAGRAPH)
    assert ai.score > human.score
    assert ai.score >= 0.55


def test_long_article_keeps_fences(settings: Settings) -> None:
    text = repeat_article(12)
    result = _pipe(settings).run(text, mode=PipelineMode.CLEAN)
    assert "KEEP_FENCE = True" in result.output
    assert result.output.count("```") >= 2
    assert_pipeline_sane(result)


def test_similarity_identity_is_high() -> None:
    assert tfidf_cosine(AI_PARAGRAPH, AI_PARAGRAPH) > 0.99
