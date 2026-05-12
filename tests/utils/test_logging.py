"""Tests for whimstan.utils.logging."""

from __future__ import annotations

import logging


def test_get_logger_returns_namespaced_logger() -> None:
    from whimstan.utils.logging import get_logger

    log = get_logger("forward")
    assert isinstance(log, logging.Logger)
    assert log.name == "whimstan.forward"


def test_setup_logging_idempotent() -> None:
    """Repeated setup_logging calls must not stack handlers."""
    from whimstan.utils.logging import setup_logging

    log = setup_logging(level="INFO")
    n_handlers = len(log.handlers)
    setup_logging(level="INFO")
    assert len(log.handlers) == n_handlers
