from __future__ import annotations

from unsynth.eval.report import compare_texts, render_markdown_report
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.rewriters.backtranslate import BacktranslateRewriter
from unsynth.rewriters.paraphrase import ParaphraseRewriter
from unsynth.types import PipelineMode


def test_compare_texts_and_report(ai_essay: str, settings) -> None:  # type: ignore[no-untyped-def]
    rewritten = ai_essay.replace("leverage", "use").replace("Furthermore", "Also")
    report = compare_texts(ai_essay, rewritten, settings=settings)
    assert report.before.score >= 0.0
    assert report.after is not None
    assert report.quality is not None
    assert report.token_stats is not None
    assert report.token_stats.original_tokens > 0


def test_markdown_report_contains_honesty(ai_essay: str, settings) -> None:  # type: ignore[no-untyped-def]
    result = UnSynthPipeline(settings).run(ai_essay, mode=PipelineMode.DETECT)
    md = render_markdown_report(result)
    assert "UnSynth report" in md
    assert "Honesty" in md
    assert "classical" in md


def test_paraphrase_and_backtranslate_skip_without_backend(settings) -> None:  # type: ignore[no-untyped-def]
    text = "A short passage that should not be sent anywhere."
    para = ParaphraseRewriter(settings).rewrite(text, strength=0.5)
    assert para.rewritten == text
    assert "backend-unavailable" in para.notes
    back = BacktranslateRewriter(settings).rewrite(text, strength=0.5)
    assert back.rewritten == text
    assert "disabled" in back.notes
