"""Score a published article without rewriting it."""

from __future__ import annotations

from pathlib import Path

from unsynth.pipeline.orchestrator import UnSynthPipeline

SAMPLE = Path(__file__).with_name("sample_article.md")


def main() -> None:
    text = SAMPLE.read_text(encoding="utf-8")
    result = UnSynthPipeline().detect(text)
    print(f"{result.name}: {result.score:.3f} ({result.label}, conf={result.confidence:.2f})")
    members = result.details.get("members") if isinstance(result.details, dict) else None
    if isinstance(members, list):
        for m in members:
            print(f"  {m['name']:14} {m['score']:.3f}  {m['label']}")


if __name__ == "__main__":
    main()
