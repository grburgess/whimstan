"""Tests for Dataset container."""

from __future__ import annotations

from pathlib import Path


def _one_obs(swift_xrt_dir: Path, name: str = "apc"):
    from whimstan.data import load_swift_xrt_observation

    return load_swift_xrt_observation(
        name=name,
        pha_path=swift_xrt_dir / "apc.pi",
        rmf_path=swift_xrt_dir / "apc.rmf",
        arf_path=swift_xrt_dir / "apc.arf",
        bkg_path=swift_xrt_dir / "apcback.pi",
        z=2.0,
        nh_mw=1.0e20,
        ra_deg=0.0,
        dec_deg=0.0,
    )


def test_dataset_construction(swift_xrt_dir: Path) -> None:
    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs, obs])
    assert ds.n_grbs == 3
    assert ds.n_ene == obs.static.n_ene
    assert ds.n_chan == obs.static.n_chan


def test_dataset_stack_static(swift_xrt_dir: Path) -> None:
    """Stacking ObservationStatic must broadcast over GRB axis."""
    import jax.numpy as jnp

    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs])
    static = ds.stack_static()

    assert static.rmf.shape == (2, obs.static.n_chan, obs.static.n_ene)
    assert static.arf.shape == (2, obs.static.n_ene)
    assert static.exposure.shape == (2,)
    assert static.z.shape == (2,)
    assert jnp.allclose(static.z, jnp.array([obs.static.z, obs.static.z]))


def test_dataset_total_counts(swift_xrt_dir: Path) -> None:
    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs])
    totals = ds.total_counts()
    assert totals.shape == (2,)
    assert int(totals[0]) == int(obs.counts.sum())
