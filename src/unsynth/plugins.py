"""Entry-point + directory plugin loader for detectors and rewriters."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from unsynth.exceptions import PluginError
from unsynth.logging import get_logger

log = get_logger("plugins")

T = TypeVar("T")

DETECTOR_GROUP = "unsynth.detectors"
REWRITER_GROUP = "unsynth.rewriters"


def _iter_entry_points(group: str) -> Iterable[importlib.metadata.EntryPoint]:
    try:
        return importlib.metadata.entry_points(group=group)
    except TypeError:  # pragma: no cover - py3.11 compat path is the default
        eps = importlib.metadata.entry_points()
        return eps.select(group=group) if hasattr(eps, "select") else ()


def load_entry_points(group: str) -> dict[str, type[object]]:
    found: dict[str, type[object]] = {}
    for ep in _iter_entry_points(group):
        try:
            loaded = ep.load()
        except Exception as exc:
            log.warning("skipping plugin %s: %s", ep.name, exc)
            continue
        if not isinstance(loaded, type):
            log.warning("plugin %s is not a class", ep.name)
            continue
        found[ep.name] = loaded
    return found


def load_directory_plugins(directories: Iterable[str | Path]) -> dict[str, type[object]]:
    """Import ``*.py`` files from extra plugin directories.

    Each module may expose ``DETECTORS`` or ``REWRITERS`` lists of classes,
    or a single ``Plugin`` class.
    """

    found: dict[str, type[object]] = {}
    for raw in directories:
        directory = Path(raw).expanduser()
        if not directory.is_dir():
            raise PluginError(f"plugin directory does not exist: {directory}")
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"unsynth_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise PluginError(f"failed to import plugin {path}: {exc}") from exc
            classes: list[type[object]] = []
            for attr in ("DETECTORS", "REWRITERS", "PLUGINS"):
                extra = getattr(module, attr, None)
                if extra:
                    classes.extend(list(extra))
            single = getattr(module, "Plugin", None)
            if isinstance(single, type):
                classes.append(single)
            for cls in classes:
                name = str(getattr(cls, "name", cls.__name__)).lower()
                found[name] = cls
    return found
