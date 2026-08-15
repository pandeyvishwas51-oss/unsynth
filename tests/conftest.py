"""Shared fixtures."""

from __future__ import annotations

import pytest

from unsynth.config import Settings

AI_ESSAY = """
In today's digital age, it is important to note that artificial intelligence
plays a crucial role in the modern landscape. Furthermore, organizations can
leverage cutting-edge models in order to streamline their comprehensive
workflows. Moreover, a wide range of stakeholders should utilize these robust
tools to unlock transformative value. Additionally, studies have shown that
teams who delve into the data subsequently enhance outcomes. In conclusion,
the key takeaway is that leveraging AI will revolutionize the industry.
""".strip()

HUMAN_NOTE = """
I keep a paper notebook next to the kettle. This morning the dog stole a
sock and I spent ten minutes fishing it out from under the radiator — which
is to say, not a productive start. The draft I promised you is late. Sorry.
I'll send the middle section tonight if the wifi holds. Three numbers I
double-checked: 14, 2.5%, and the April invoice. Don't publish the names.
""".strip()


@pytest.fixture()
def settings() -> Settings:
    cfg = Settings()
    cfg.backend.kind = cfg.backend.kind.NONE
    cfg.rewrite.max_passes = 2
    cfg.rewrite.initial_strength = 0.55
    cfg.quality.embeddings = "none"
    return cfg


@pytest.fixture()
def ai_essay() -> str:
    return AI_ESSAY


@pytest.fixture()
def human_note() -> str:
    return HUMAN_NOTE
