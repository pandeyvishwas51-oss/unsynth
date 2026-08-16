"""Named detector factory."""

from __future__ import annotations

from collections.abc import Iterable

from unsynth.config import Settings
from unsynth.detectors.anthropic import AnthropicDetectionAdapter
from unsynth.detectors.base import BaseDetector
from unsynth.detectors.classical import ClassicalDetector
from unsynth.detectors.statistical import StatisticalWatermarkDetector
from unsynth.detectors.stylometric import StylometricDetector
from unsynth.exceptions import PluginError
from unsynth.logging import get_logger
from unsynth.plugins import DETECTOR_GROUP, load_directory_plugins, load_entry_points

log = get_logger("detectors.registry")


class DetectorRegistry:
    """Built-ins plus setuptools entry points plus extra directories."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._classes: dict[str, type[BaseDetector]] = {
            "classical": ClassicalDetector,
            "stylometric": StylometricDetector,
            "statistical": StatisticalWatermarkDetector,
            "anthropic": AnthropicDetectionAdapter,
        }
        for name, cls in load_entry_points(DETECTOR_GROUP).items():
            if isinstance(cls, type) and issubclass(cls, BaseDetector):
                self._classes.setdefault(name, cls)
        extra = load_directory_plugins(self.settings.runtime.plugin_dirs)
        for name, cls in extra.items():
            if isinstance(cls, type) and issubclass(cls, BaseDetector):
                self._classes[name] = cls

    def available(self) -> list[str]:
        return sorted(self._classes)

    def create(self, name: str) -> BaseDetector:
        key = name.lower().strip()
        if key not in self._classes:
            raise PluginError(
                f"unknown detector {name!r}; available: {', '.join(self.available())}"
            )
        return self._classes[key](self.settings)

    def create_many(self, names: Iterable[str] | None = None) -> list[BaseDetector]:
        wanted = list(names) if names is not None else list(self.settings.detect.detectors)
        out: list[BaseDetector] = []
        for name in wanted:
            try:
                out.append(self.create(name))
            except PluginError:
                log.warning("skipping unknown detector %s", name)
        return out
