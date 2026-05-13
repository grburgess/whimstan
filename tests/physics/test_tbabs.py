"""Tests for TbAbs JAX cross section."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_sigma_loaded() -> None:
    from whimstan.physics.tbabs import TBABS_E_KEV, TBABS_SIGMA_CM2

    assert TBABS_E_KEV.shape == TBABS_SIGMA_CM2.shape
    assert TBABS_E_KEV.shape[0] >= 1024
    assert float(TBABS_SIGMA_CM2.min()) > 0
    assert float(TBABS_SIGMA_CM2.max()) < 1e-18  # sanity: < 1e-18 cm^2/H


def test_sigma_decreases_with_E() -> None:
    """Above all absorption edges (E > 10 keV), sigma drops monotonically.

    Note: the 0.1-10 keV band contains real photoelectric edges (e.g. the
    Fe K edge at ~7.1 keV) so monotonicity holds only above ~10 keV with
    the real Wilms+ 2000 TbAbs table.
    """
    from whimstan.physics.tbabs import tbabs_sigma

    E_high = jnp.linspace(10.0, 100.0, 50)
    s = tbabs_sigma(E_high)
    assert jnp.all(jnp.diff(s) <= 0), "sigma must decrease above 10 keV"


def test_sigma_at_known_energy() -> None:
    """Approximate sigma at 1 keV (~few x 1e-22 cm^2 per H for solar TbAbs)."""
    from whimstan.physics.tbabs import tbabs_sigma

    s_1kev = float(tbabs_sigma(jnp.array(1.0)))
    assert 1e-22 < s_1kev < 1e-21, f"sigma(1 keV)={s_1kev:.3e} outside expected range"


def test_jit_and_grad() -> None:
    from whimstan.physics.tbabs import tbabs_sigma

    f = jax.jit(tbabs_sigma)
    E = jnp.linspace(0.4, 10.0, 50)
    s = f(E)
    assert s.shape == E.shape
    g = jax.grad(lambda E_: tbabs_sigma(E_).sum())(E)
    assert jnp.all(jnp.isfinite(g))


def test_e_below_grid_clipped() -> None:
    """Below the table E_lo, return the value at E_lo (clip; do not extrapolate)."""
    from whimstan.physics.tbabs import TBABS_E_KEV, tbabs_sigma

    s_below = tbabs_sigma(jnp.array(TBABS_E_KEV[0] / 2))
    s_lo = tbabs_sigma(jnp.array(TBABS_E_KEV[0]))
    assert pytest.approx(float(s_below), rel=1e-6) == float(s_lo)
