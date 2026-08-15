"""Shared corpora and builders for production / stress tests."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

from unsynth.types import DetectorResult, PipelineResult

AI_PARAGRAPH = (
    "In today's digital age, it is important to note that artificial intelligence "
    "plays a crucial role in the modern landscape. Furthermore, organizations can "
    "leverage cutting-edge models in order to streamline their comprehensive "
    "workflows. Moreover, a wide range of stakeholders should utilize these robust "
    "tools to unlock transformative value. Additionally, studies have shown that "
    "teams who delve into the data subsequently enhance outcomes. In conclusion, "
    "the key takeaway is that leveraging AI will revolutionize the industry."
)

HUMAN_PARAGRAPH = (
    "I keep a paper notebook next to the kettle. This morning the dog stole a "
    "sock and I spent ten minutes fishing it out from under the radiator — which "
    "is to say, not a productive start. The draft I promised you is late. Sorry. "
    "I'll send the middle section tonight if the wifi holds. Three numbers I "
    "double-checked: 14, 2.5%, and the April invoice. Don't publish the names."
)

MARKDOWN_DOC = """\
---
title: keep-me
generator: should-stay-in-frontmatter
---

# Heading Must Survive

Intro paragraph about nothing much.

```python
SECRET_TOKEN = "do-not-touch"
def keep():
    return 42
```

| metric | value |
| --- | --- |
| kept | yes |

Visit https://example.com/path?q=1 and mail ops@example.com for the dump.

- list item one
- list item two

> a quoted line that can change a bit
"""


def repeat_article(paragraphs: int, *, ai: bool = True) -> str:
    block = AI_PARAGRAPH if ai else HUMAN_PARAGRAPH
    chunks = [f"# Article {i}\n\n{block} ({i}.)\n" for i in range(paragraphs)]
    chunks.insert(2, "```\nKEEP_FENCE = True\n```\n")
    return "\n".join(chunks)


def mixed_unicode_doc() -> str:
    return (
        "Café naïve façade. Москва уже проснулась. 東京は雨です。\n\n"
        "It is important to note that we leverage robust tools.\n"
        "Emoji check 😀👍 and a zero-width\u200bspace hide.\n"
        "https://exámple.invalid/café should stay put.\n"
    )


def assert_unit(value: float, name: str = "score") -> None:
    assert isinstance(value, (int, float)), f"{name} not numeric: {value!r}"
    assert math.isfinite(value), f"{name} not finite: {value!r}"
    assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def assert_result_sane(result: DetectorResult) -> None:
    assert_unit(result.score, "score")
    assert_unit(result.confidence, "confidence")
    assert result.label in {"human", "ai", "watermarked", "uncertain"}
    payload = result.as_dict()
    json.dumps(payload)  # must be serializable
    for signal in result.signals:
        assert math.isfinite(signal.value)


def assert_pipeline_sane(result: PipelineResult) -> None:
    assert_result_sane(result.before)
    if result.after is not None:
        assert_result_sane(result.after)
    assert isinstance(result.output, str)
    json.dumps(result.as_dict())


def assert_protected_intact(original: str, rewritten: str, needles: Iterable[str]) -> None:
    for needle in needles:
        if needle in original:
            assert needle in rewritten, f"protected span lost: {needle!r}"
