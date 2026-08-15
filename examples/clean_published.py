"""Clean an already-published blog post and print the before/after report."""

from __future__ import annotations

from pathlib import Path

from unsynth.eval.report import render_markdown_report
from unsynth.pipeline.orchestrator import run_pipeline
from unsynth.types import PipelineMode

SAMPLE = Path(__file__).with_name("sample_article.md")


def main() -> None:
    text = SAMPLE.read_text(encoding="utf-8")
    result = run_pipeline(text, mode=PipelineMode.CLEAN)
    print(render_markdown_report(result))
    print("\n----- cleaned -----\n")
    print(result.output)


if __name__ == "__main__":
    main()
