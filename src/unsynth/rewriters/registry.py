"""Named rewriter factory."""

from __future__ import annotations

from collections.abc import Iterable

from unsynth.config import Settings
from unsynth.exceptions import PluginError
from unsynth.plugins import REWRITER_GROUP, load_directory_plugins, load_entry_points
from unsynth.rewriters.backtranslate import BacktranslateRewriter
from unsynth.rewriters.base import BaseRewriter
from unsynth.rewriters.lexical import LexicalRewriter
from unsynth.rewriters.paraphrase import ParaphraseRewriter
from unsynth.rewriters.structural import StructuralRewriter
from unsynth.rewriters.style import StyleHumanizer


class RewriterRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._classes: dict[str, type[BaseRewriter]] = {
            "lexical": LexicalRewriter,
            "structural": StructuralRewriter,
            "style": StyleHumanizer,
            "paraphrase": ParaphraseRewriter,
            "backtranslate": BacktranslateRewriter,
        }
        for name, cls in load_entry_points(REWRITER_GROUP).items():
            if isinstance(cls, type) and issubclass(cls, BaseRewriter):
                self._classes.setdefault(name, cls)
        for name, cls in load_directory_plugins(self.settings.runtime.plugin_dirs).items():
            if isinstance(cls, type) and issubclass(cls, BaseRewriter):
                self._classes[name] = cls

    def available(self) -> list[str]:
        return sorted(self._classes)

    def create(self, name: str) -> BaseRewriter:
        key = name.lower().strip()
        if key not in self._classes:
            raise PluginError(
                f"unknown rewriter {name!r}; available: {', '.join(self.available())}"
            )
        return self._classes[key](self.settings)

    def create_many(self, names: Iterable[str] | None = None) -> list[BaseRewriter]:
        wanted = list(names) if names is not None else list(self.settings.rewrite.strategies)
        return [self.create(n) for n in wanted]
