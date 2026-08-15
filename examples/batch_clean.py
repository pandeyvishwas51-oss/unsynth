"""Batch-clean every markdown file under a directory."""

from __future__ import annotations

import sys
from pathlib import Path

from unsynth.pipeline.document import iter_files
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out")
    dest.mkdir(parents=True, exist_ok=True)
    pipeline = UnSynthPipeline()
    for path in iter_files([root]):
        result = pipeline.run(path.read_text(encoding="utf-8"), mode=PipelineMode.CLEAN)
        target = dest / (path.stem + ".clean.md")
        target.write_text(result.output, encoding="utf-8")
        after = result.after.score if result.after else result.before.score
        print(f"{path}  {result.before.score:.3f} → {after:.3f}  → {target}")


if __name__ == "__main__":
    main()
