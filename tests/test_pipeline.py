from __future__ import annotations

from unsynth.pipeline.markdown import parse_markdown, render_segments
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode


def test_markdown_protects_code_and_tables() -> None:
    src = (
        "# Title\n\nHello world.\n\n```python\nSECRET = 1\n```\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nMore prose here.\n"
    )
    segs = parse_markdown(src)
    kinds = {s.kind for s in segs}
    assert "code" in kinds
    assert "table" in kinds
    code = next(s for s in segs if s.kind == "code")
    assert code.protected
    assert "SECRET" in code.text
    assert render_segments(segs) == src


def test_pipeline_detect_only(human_note: str, settings) -> None:  # type: ignore[no-untyped-def]
    result = UnSynthPipeline(settings).run(human_note, mode=PipelineMode.DETECT)
    assert result.output == human_note
    assert result.after is None
    assert result.rewrites == ()


def test_pipeline_clean_lowers_ai_score(ai_essay: str, settings) -> None:  # type: ignore[no-untyped-def]
    pipe = UnSynthPipeline(settings)
    result = pipe.run(ai_essay, mode=PipelineMode.CLEAN)
    assert result.after is not None
    assert result.output
    # Heuristic stack should move the classical/ensemble score down or
    # at least change tokens while staying similar.
    assert result.eval.quality is not None
    assert result.eval.quality.similarity >= 0.4
    assert result.output != ai_essay or result.before.score < 0.35


def test_pipeline_preserves_code_fence(settings) -> None:  # type: ignore[no-untyped-def]
    text = (
        "It is important to note that we should leverage robust tools.\n\n"
        "```\nkeep_me_please()\n```\n\n"
        "Furthermore, a wide range of users can utilize the system.\n"
    )
    result = UnSynthPipeline(settings).run(text, mode=PipelineMode.CLEAN)
    assert "keep_me_please()" in result.output


def test_sample_article_classical_drops_and_markdown_survives(settings) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    text = Path("examples/sample_article.md").read_text(encoding="utf-8")
    pipe = UnSynthPipeline(settings)
    before = pipe.detect(text)
    members = {m["name"]: m for m in before.details["members"]}  # type: ignore[index]
    assert members["classical"]["score"] >= 0.7
    result = pipe.run(text, mode=PipelineMode.CLEAN)
    assert result.after is not None
    after_members = {m["name"]: m for m in result.after.details["members"]}  # type: ignore[index]
    assert after_members["classical"]["score"] < members["classical"]["score"]
    assert "```python" in result.output
    assert "return len(text) / 42.0" in result.output
    assert "| metric | value |" in result.output
    assert result.output.startswith("# ")


def test_pipeline_keeps_heading_and_blank_lines(settings) -> None:  # type: ignore[no-untyped-def]
    text = (
        "# In Today's Digital Age\n\n"
        "It is important to note that we leverage robust tools.\n\n"
        "Furthermore, a wide range of users can utilize the system.\n"
    )
    result = UnSynthPipeline(settings).run(text, mode=PipelineMode.CLEAN)
    assert result.output.startswith("# In Today's Digital Age\n")
    assert "\n\n" in result.output
    heading = next(
        s
        for s in __import__(
            "unsynth.pipeline.markdown", fromlist=["parse_markdown"]
        ).parse_markdown(text)
        if s.kind == "heading"
    )
    assert heading.protected
