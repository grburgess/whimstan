"""OGIP data loaders and Observation containers."""

from __future__ import annotations

from whimstan.data.catalog import (
    CatalogEntry,
    filter_catalog,
    load_catalog,
    load_packaged_catalog,
)
from whimstan.data.ogip import load_swift_xrt_observation
from whimstan.data.types import Observation, ObservationStatic

__all__ = [
    "CatalogEntry",
    "Observation",
    "ObservationStatic",
    "filter_catalog",
    "load_catalog",
    "load_packaged_catalog",
    "load_swift_xrt_observation",
]
