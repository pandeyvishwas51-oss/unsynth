"""Language-model backends. Local by default; cloud only if you ask."""

from unsynth.backends.base import complete, is_available

__all__ = ["complete", "is_available"]
