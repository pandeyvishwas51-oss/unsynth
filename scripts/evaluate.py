#!/usr/bin/env python3
"""Batch evaluation harness.

Usage:
    uv run python scripts/evaluate.py examples/sample_article.md
    uv run python scripts/evaluate.py --dir corpus/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsynth.config import load_settings
from unsynth.eval.report import render_markdown_report
from unsynth.pipeline.document import iter_files
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode


def main() -> None:
    parser = argparse.ArgumentParser(description="UnSynth evaluation harness")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    pipeline = UnSynthPipeline(settings)
    files = list(args.paths)
    if args.dir:
        files.extend(iter_files([args.dir]))
    if not files:
        files = [Path("examples/sample_article.md")]

    rows = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        result = pipeline.run(text, mode=PipelineMode.CLEAN)
        after = result.after.score if result.after else None
        rows.append(
            {
                "path": str(path),
                "before": result.before.score,
                "after": after,
                "delta": (None if after is None else result.before.score - after),
                "similarity": result.eval.quality.similarity if result.eval.quality else None,
                "target_met": result.target_met,
                "passes": result.eval.passes,
            }
        )
        if not args.json:
            print(render_markdown_report(result))
            print("-" * 60)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"evaluated {len(rows)} file(s)")


if __name__ == "__main__":
    main()
