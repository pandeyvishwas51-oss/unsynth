"""Multi-pass paraphraser. Uses a local LLM backend when configured.

If no backend is available this rewriter is a no-op (the heuristic layers
already ran). That is intentional: UnSynth stays useful offline and never
silently phones a cloud API.
"""

from __future__ import annotations

from unsynth.backends import complete, is_available
from unsynth.rewriters.base import BaseRewriter
from unsynth.types import RewriteResult

SYSTEM = (
    "You rewrite English so it keeps the same facts, names, numbers, and "
    "links, but changes wording and sentence shape. Prefer contractions, "
    "uneven sentence length, and concrete verbs. Avoid stock phrases such as "
    "'delve', 'leverage', 'landscape', 'robust', 'in conclusion', "
    "'it is important to note'. Do not add facts. Do not explain. "
    "Return only the rewritten text."
)

PROMPT = """Rewrite the passage below.
Strength: {strength:.2f} (0 = light touch, 1 = aggressive restructure).
Keep meaning. Change local word choice and clause order.
Do not wrap the answer in quotes or markdown fences.

PASSAGE:
{text}
"""


class ParaphraseRewriter(BaseRewriter):
    name = "paraphrase"
    requires_backend = True

    def rewrite(self, text: str, *, strength: float = 0.45) -> RewriteResult:
        if not is_available(self.settings):
            return self.pack(
                text,
                text,
                strength=strength,
                edits=0,
                notes=("backend-unavailable",),
            )
        prompt = PROMPT.format(strength=max(0.0, min(1.0, strength)), text=text.strip())
        temperature = min(1.15, self.settings.backend.temperature + 0.25 * strength)
        try:
            out = complete(self.settings, prompt, system=SYSTEM, temperature=temperature)
        except Exception as exc:  # backend errors should not kill the stack
            return self.pack(
                text,
                text,
                strength=strength,
                edits=0,
                notes=(f"backend-error:{exc}",),
            )
        cleaned = _strip_fences(out).strip()
        if not cleaned:
            return self.pack(text, text, strength=strength, edits=0, notes=("empty-backend",))
        return self.pack(text, cleaned, strength=strength, edits=1, notes=("llm",))


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
