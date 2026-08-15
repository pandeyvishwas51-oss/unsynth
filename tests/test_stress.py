"""Scale and load tests. Default ones stay CI-fast; @slow is the heavy belt."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from tests.helpers import (
    AI_PARAGRAPH,
    HUMAN_PARAGRAPH,
    assert_pipeline_sane,
    repeat_article,
)

from unsynth.config import Settings
from unsynth.pipeline.document import iter_files
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode


def test_medium_document_budget(settings: Settings) -> None:
    """~8–12 KB mixed markdown must finish in well under a few seconds."""

    text = repeat_article(25)
    assert len(text) > 8_000
    pipe = UnSynthPipeline(settings)
    start = time.perf_counter()
    result = pipe.run(text, mode=PipelineMode.CLEAN)
    elapsed = time.perf_counter() - start
    assert_pipeline_sane(result)
    assert "KEEP_FENCE = True" in result.output
    assert elapsed < 8.0, f"medium document took {elapsed:.2f}s"


def test_batch_directory_isolation(settings: Settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "in"
    dest = tmp_path / "out"
    src.mkdir()
    dest.mkdir()
    for i in range(12):
        body = AI_PARAGRAPH if i % 2 == 0 else HUMAN_PARAGRAPH
        (src / f"doc-{i:02d}.md").write_text(f"# D{i}\n\n{body}\n", encoding="utf-8")
    (src / "ignore.bin").write_bytes(b"\x00\x01\x02")
    files = list(iter_files([src]))
    assert len(files) == 12
    pipe = UnSynthPipeline(settings)
    for path in files:
        result = pipe.run(path.read_text(encoding="utf-8"), mode=PipelineMode.CLEAN)
        assert_pipeline_sane(result)
        target = dest / path.name
        target.write_text(result.output, encoding="utf-8")
    assert len(list(dest.glob("*.md"))) == 12


def test_many_passes_do_not_explode(settings: Settings) -> None:
    settings.rewrite.max_passes = 5
    settings.rewrite.initial_strength = 0.3
    settings.rewrite.strength_step = 0.15
    result = UnSynthPipeline(settings).run(AI_PARAGRAPH * 2, mode=PipelineMode.CLEAN)
    assert_pipeline_sane(result)
    assert result.eval.passes <= 5
    if result.eval.quality is not None:
        assert result.eval.quality.length_ratio > 0.2


def test_thread_pool_detect_is_safe(settings: Settings) -> None:
    pipe = UnSynthPipeline(settings)
    corpus = [AI_PARAGRAPH, HUMAN_PARAGRAPH, AI_PARAGRAPH[::-1], HUMAN_PARAGRAPH * 2] * 4

    def _one(text: str) -> float:
        return pipe.detect(text).score

    with ThreadPoolExecutor(max_workers=8) as pool:
        scores = list(pool.map(_one, corpus))
    assert len(scores) == len(corpus)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.stress
@pytest.mark.slow
def test_large_document_50kb(settings: Settings) -> None:
    text = repeat_article(160)
    assert len(text) > 50_000
    settings.rewrite.max_passes = 1
    start = time.perf_counter()
    result = UnSynthPipeline(settings).run(text, mode=PipelineMode.REWRITE)
    elapsed = time.perf_counter() - start
    assert_pipeline_sane(result)
    assert "KEEP_FENCE = True" in result.output
    assert elapsed < 25.0, f"50KB rewrite took {elapsed:.2f}s"


@pytest.mark.stress
@pytest.mark.slow
def test_batch_40_files(settings: Settings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings.rewrite.max_passes = 1
    pipe = UnSynthPipeline(settings)
    for i in range(40):
        (tmp_path / f"{i}.md").write_text(AI_PARAGRAPH + f" ({i})\n", encoding="utf-8")
    start = time.perf_counter()
    for path in sorted(tmp_path.glob("*.md")):
        result = pipe.run(path.read_text(encoding="utf-8"), mode=PipelineMode.REWRITE)
        assert_pipeline_sane(result)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"40-file batch took {elapsed:.2f}s"
