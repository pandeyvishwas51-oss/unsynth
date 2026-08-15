"""Randomized property tests.  No extra deps — stdlib random, fixed seed."""

from __future__ import annotations

import random
import string

from tests.helpers import AI_PARAGRAPH, assert_pipeline_sane, assert_result_sane

from unsynth.config import Settings
from unsynth.pipeline.markdown import parse_markdown, render_segments
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.rewriters.lexical import LexicalRewriter
from unsynth.rewriters.quality import QualityGate
from unsynth.rewriters.structural import StructuralRewriter
from unsynth.rewriters.style import StyleHumanizer
from unsynth.types import PipelineMode

ALPHABET = string.ascii_letters + string.digits + " .,!?;:'\"-\n#*`|[]()/"


def _noise(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def test_fuzz_detect_500_random_strings(settings: Settings) -> None:
    rng = random.Random(20260815)
    pipe = UnSynthPipeline(settings)
    for _ in range(500):
        n = rng.randint(0, 400)
        blob = _noise(rng, n)
        if rng.random() < 0.2:
            blob = blob + "\x00" + AI_PARAGRAPH[: rng.randint(0, 80)]
        assert_result_sane(pipe.detect(blob))


def test_fuzz_rewrite_strategies(settings: Settings) -> None:
    rng = random.Random(7)
    strategies = [
        LexicalRewriter(settings),
        StructuralRewriter(settings),
        StyleHumanizer(settings),
    ]
    for i in range(80):
        blob = _noise(rng, rng.randint(20, 300))
        if i % 3 == 0:
            blob = AI_PARAGRAPH[: rng.randint(40, len(AI_PARAGRAPH))] + " " + blob
        for strat in strategies:
            out = strat.rewrite(blob, strength=rng.choice([0.1, 0.5, 0.9]))
            assert isinstance(out.rewritten, str)


def test_fuzz_markdown_roundtrip() -> None:
    rng = random.Random(99)
    for _ in range(60):
        lines = []
        for _line in range(rng.randint(3, 25)):
            kind = rng.choice(["p", "h", "code", "table", "blank", "list"])
            if kind == "p":
                lines.append(_noise(rng, rng.randint(8, 80)))
            elif kind == "h":
                lines.append("# " + _noise(rng, 12))
            elif kind == "code":
                lines.append("```")
                lines.append(_noise(rng, 20))
                lines.append("```")
            elif kind == "table":
                lines.append("| a | b |")
                lines.append("| --- | --- |")
                lines.append("| 1 | 2 |")
            elif kind == "list":
                lines.append("- " + _noise(rng, 16))
            else:
                lines.append("")
        src = "\n".join(lines) + "\n"
        assert render_segments(parse_markdown(src)) == src


def test_fuzz_clean_mixed_docs(settings: Settings) -> None:
    rng = random.Random(123)
    settings.rewrite.max_passes = 1
    pipe = UnSynthPipeline(settings)
    for _ in range(25):
        body = "\n\n".join(
            rng.choice([AI_PARAGRAPH, _noise(rng, 80), "```\nX=1\n```"]) for _k in range(5)
        )
        result = pipe.run(body, mode=PipelineMode.REWRITE)
        assert_pipeline_sane(result)
        if "```\nX=1\n```" in body:
            assert "X=1" in result.output


def test_quality_gate_never_nans(settings: Settings) -> None:
    rng = random.Random(3)
    gate = QualityGate(settings)
    for _ in range(40):
        a = _noise(rng, rng.randint(0, 200))
        b = _noise(rng, rng.randint(0, 200))
        report = gate.evaluate(a, b)
        assert 0.0 <= report.similarity <= 1.0
        assert report.length_ratio >= 0.0
