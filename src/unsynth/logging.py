"""Small logging helpers. CLI attaches a Rich handler."""

from __future__ import annotations

import logging

LOGGER_NAME = "unsynth"


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: str = "INFO") -> None:
    logger = get_logger()
    if logger.handlers:
        logger.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
