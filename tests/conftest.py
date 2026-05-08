"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

# Importing whimstan early ensures jax x64 is enabled before any other
# fixture or test code creates JAX arrays.
import whimstan  # noqa: F401


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def swift_xrt_dir(fixtures_dir: Path) -> Path:
    d = fixtures_dir / "swift_xrt"
    assert d.is_dir(), d
    return d
