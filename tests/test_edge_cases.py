"""Adversarial and weird inputs that production traffic will throw at us."""

from __future__ import annotations

import json

import pytest
from tests.helpers import AI_PARAGRAPH, assert_pipeline_sane, assert_result_sane, mixed_unicode_doc

from unsynth.config import Settings
from unsynth.detectors.ensemble import EnsembleDetector
from unsynth.metadata.strip import strip_html, strip_path
from unsynth.pipeline.markdown import parse_markdown, render_segments
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.rewriters.lexical import LexicalRewriter
from unsynth.safety import sanitize_text
from unsynth.types import PipelineMode


@pytest.mark.parametrize(
    "blob",
    [
        "",
        " ",
        "\n\n\n",
        "\t",
        "a",
        "???",
        "12345 67890 11111",
        "🎉" * 50,
        "\u200b\u200c\u200d word " * 8,
        "A" * 5000,
        "the " * 200,
        "Hello.\n" * 3,
        "https://only-a-url.example/x",
        "ops@example.com " * 6,
        "```\nunclosed fence\n",
        "~~~\nalso a fence\n~~~",
        "```python:3.11\nprint(1)\n```",
        "---\nonly-open-frontmatter\n",
        "# only heading\n",
        "| a | b |\n| --- | --- |\n",
        "<div>html line</div>\nprose here that can change a little bit more.\n",
        mixed_unicode_doc(),
        "\x00NUL" + AI_PARAGRAPH,
        "leverage " * 40,
    ],
)
def test_pipeline_never_crashes_on_garbage(settings: Settings, blob: str) -> None:
    result = UnSynthPipeline(settings).run(blob, mode=PipelineMode.CLEAN)
    assert_pipeline_sane(result)


def test_unclosed_fence_stays_protected(settings: Settings) -> None:
    text = "Please leverage robust tools.\n\n```python\nSECRET = 1\n"
    segs = parse_markdown(text)
    assert any(s.kind == "code" and s.protected for s in segs)
    result = UnSynthPipeline(settings).run(text, mode=PipelineMode.CLEAN)
    assert "SECRET = 1" in result.output


def test_tilde_and_infostring_fences() -> None:
    src = "Intro text here.\n\n~~~rust\nlet x = 1;\n~~~\n\n```python:3.11\nprint(1)\n```\n"
    assert render_segments(parse_markdown(src)) == src
    code = [s for s in parse_markdown(src) if s.kind == "code"]
    assert len(code) == 2
    assert all(c.protected for c in code)


def test_urls_not_rewritten(settings: Settings) -> None:
    text = (
        "It is important to note that we leverage robust systems. "
        "Docs live at https://api.example.com/v1/leverage and "
        "support is ops@leverage.example."
    )
    out = LexicalRewriter(settings).rewrite(text, strength=0.95)
    assert "https://api.example.com/v1/leverage" in out.rewritten
    assert "ops@leverage.example" in out.rewritten


def test_word_boundary_does_not_nibble_android(settings: Settings) -> None:
    text = (
        "The android landed. Furthermore the landscape of the band was quiet "
        "and the candidates were utilizing notes."
    )
    out = LexicalRewriter(settings).rewrite(text, strength=0.99)
    assert "android" in out.rewritten.lower()


def test_html_strip_does_not_eat_body() -> None:
    html = (
        "<html><head>"
        "<meta name='generator' content='ChatGPT'>"
        "<meta property='og:title' content='Hello'>"
        "</head><body><p>Keep me &amp; mine</p></body></html>"
    )
    cleaned, hits = strip_html(html)
    assert "Keep me" in cleaned
    assert "og:title" in cleaned
    assert hits


def test_strip_missing_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from unsynth.exceptions import MetadataError

    with pytest.raises(MetadataError):
        strip_path(tmp_path / "nope.txt")


def test_ensemble_detect_windows_short_and_long(settings: Settings) -> None:
    ens = EnsembleDetector(settings)
    short = ens.detect_windows("tiny")
    assert short
    assert_result_sane(short[0])
    long = ens.detect_windows(AI_PARAGRAPH * 8, window_tokens=40, stride=20)
    assert len(long) >= 2
    for item in long:
        assert_result_sane(item)


def test_nul_input_is_sanitized_by_pipeline(settings: Settings) -> None:
    result = UnSynthPipeline(settings).run("\x00" + AI_PARAGRAPH, mode=PipelineMode.DETECT)
    assert "\x00" not in result.original


def test_sanitize_then_json(settings: Settings) -> None:
    text = sanitize_text("café\x00 " + AI_PARAGRAPH)
    result = UnSynthPipeline(settings).detect(text)
    json.dumps(result.as_dict())
