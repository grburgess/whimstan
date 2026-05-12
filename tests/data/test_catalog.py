"""Tests for GRB catalog loader."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_load_packaged_catalog() -> None:
    from whimstan.data.catalog import load_packaged_catalog

    cat = load_packaged_catalog()
    assert len(cat) >= 100
    first = cat[0]
    # name is a string and looks GRB-ish
    assert isinstance(first.name, str)
    assert len(first.name) > 0
    assert -90.0 <= first.dec_deg <= 90.0
    assert 0.0 <= first.ra_deg <= 360.0
    assert first.z > 0
    assert first.nh_mw > 0


def test_filter_by_z_and_nh() -> None:
    from whimstan.data.catalog import filter_catalog, load_packaged_catalog

    cat = load_packaged_catalog()
    n_before = len(cat)
    filtered = filter_catalog(cat, z_min=1.0, nh_mw_max=1.0e21)
    assert len(filtered) <= n_before
    for entry in filtered:
        assert entry.z >= 1.0
        assert entry.nh_mw <= 1.0e21


def test_load_missing_csv_raises(tmp_path: Path) -> None:
    from whimstan.data.catalog import load_catalog

    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.csv")
