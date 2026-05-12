"""Tests for jaxspec-backed OGIP loader."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest


def test_load_swift_xrt_observation(swift_xrt_dir: Path) -> None:
    from whimstan.data.ogip import load_swift_xrt_observation

    obs = load_swift_xrt_observation(
        name="apc",
        pha_path=swift_xrt_dir / "apc.pi",
        rmf_path=swift_xrt_dir / "apc.rmf",
        arf_path=swift_xrt_dir / "apc.arf",
        bkg_path=swift_xrt_dir / "apcback.pi",
        z=2.0,
        nh_mw=1.0e20,
        ra_deg=10.0,
        dec_deg=-20.0,
        e_lo=0.4,
        e_hi=10.0,
    )

    # Static shape sanity
    assert obs.static.n_chan > 0
    assert obs.static.n_ene > 0
    assert obs.static.rmf.shape == (obs.static.n_chan, obs.static.n_ene)
    assert obs.static.arf.shape == (obs.static.n_ene,)

    # Energy ordering
    assert jnp.all(obs.static.ene_lo < obs.static.ene_hi)
    assert jnp.all(obs.static.chan_lo < obs.static.chan_hi)

    # Channel energies inside requested band (with float tolerance)
    assert float(obs.static.chan_lo.min()) >= 0.4 - 1e-6
    assert float(obs.static.chan_hi.max()) <= 10.0 + 1e-6

    # Counts non-negative
    assert obs.counts.shape == (obs.static.n_chan,)
    assert int(obs.counts.min()) >= 0

    # Background present and non-negative
    assert obs.bkg_counts.shape == (obs.static.n_chan,)
    assert int(obs.bkg_counts.min()) >= 0

    # Per-channel bkg scale (W-stat alpha) positive
    assert obs.bkg_scale.shape == (obs.static.n_chan,)
    assert float(obs.bkg_scale.min()) > 0

    # Exposure positive
    assert float(obs.static.exposure) > 0

    # Metadata round-trip
    assert obs.name == "apc"
    assert obs.ra_deg == 10.0
    assert pytest.approx(float(obs.static.z)) == 2.0
    assert pytest.approx(float(obs.static.nh_mw)) == 1.0e20


def test_load_missing_file_raises(swift_xrt_dir: Path) -> None:
    from whimstan.data.ogip import load_swift_xrt_observation

    with pytest.raises(FileNotFoundError):
        load_swift_xrt_observation(
            name="x",
            pha_path=swift_xrt_dir / "does-not-exist.pi",
            rmf_path=swift_xrt_dir / "apc.rmf",
            arf_path=swift_xrt_dir / "apc.arf",
            bkg_path=swift_xrt_dir / "apcback.pi",
            z=1.0,
            nh_mw=1e20,
            ra_deg=0.0,
            dec_deg=0.0,
        )
