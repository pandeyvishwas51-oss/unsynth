#!/usr/bin/env python3
"""Timed production bench. Run with: uv run python scripts/stress_bench.py"""

from __future__ import annotations

import time

from unsynth.config import Settings
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode

AI_PARAGRAPH = (
    "In today's digital age, it is important to note that artificial intelligence "
    "plays a crucial role in the modern landscape. Furthermore, organizations can "
    "leverage cutting-edge models in order to streamline their comprehensive workflows."
)


def repeat_article(paragraphs: int) -> str:
    chunks = [f"# Article {i}\n\n{AI_PARAGRAPH} ({i}.)\n" for i in range(paragraphs)]
    chunks.insert(2, "```\nKEEP_FENCE = True\n```\n")
    return "\n".join(chunks)


def _time(label: str, fn) -> None:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    after = result.after.score if result.after else result.before.score
    print(
        f"{label:28} {elapsed:6.3f}s  "
        f"chars={len(result.original):6d}  "
        f"{result.before.score:.3f}→{after:.3f}  "
        f"passes={result.eval.passes}"
    )


def main() -> None:
    settings = Settings()
    settings.quality.embeddings = "none"
    settings.rewrite.max_passes = 2
    pipe = UnSynthPipeline(settings)
    _time("tiny detect", lambda: pipe.run(AI_PARAGRAPH, mode=PipelineMode.DETECT))
    _time("tiny clean", lambda: pipe.run(AI_PARAGRAPH, mode=PipelineMode.CLEAN))
    _time("12kb clean", lambda: pipe.run(repeat_article(30), mode=PipelineMode.CLEAN))
    settings.rewrite.max_passes = 1
    pipe = UnSynthPipeline(settings)
    _time("50kb rewrite", lambda: pipe.run(repeat_article(160), mode=PipelineMode.REWRITE))


if __name__ == "__main__":
    main()
