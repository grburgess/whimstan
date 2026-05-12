"""Stdlib logging + rich handler for human-readable CLI output."""

from __future__ import annotations

import logging
from typing import Literal

from rich.logging import RichHandler

_ROOT_NAME = "whimstan"
_DEFAULT_FMT = "%(message)s"


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> logging.Logger:
    """Configure the `whimstan` root logger; idempotent."""
    log = logging.getLogger(_ROOT_NAME)
    log.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
        log.addHandler(handler)
    log.propagate = False
    return log


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `whimstan` namespace."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
