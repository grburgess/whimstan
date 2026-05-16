"""Tests for the from-scratch WHIM tau validator.

Plan 1 ships an analytic stub that matches the tiny synthetic grid
within 5%; Plan 2 swaps in real Saha+photo-ionization physics.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp


def test_validator_returns_tau() -> None:
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    E_obs = jnp.logspace(jnp.log10(0.3), jnp.log10(10.0), 32)
    tau = compute_tau_from_scratch(
        log_T=6.0,
        log_n=-5.0,
        z=2.0,
        E_obs=E_obs,
    )
    assert tau.shape == E_obs.shape
    assert jnp.all(jnp.isfinite(tau))
    assert jnp.all(tau >= 0)


def test_validator_matches_tiny_grid(fixtures_dir: Path) -> None:
    """Validator must agree with shipped grid to <= 5% on overlap region."""
    from whimstan.physics.whim import WhimGrid
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    grid = WhimGrid.load(fixtures_dir / "whim_grid_tiny.h5")
    # Use grid-aligned coordinates so the trilinear interpolator returns
    # an exact stored tau slice (modulo float32->float64 cast in the grid).
    log_T = jnp.asarray(grid.log_T[2], dtype=jnp.float64)
    log_n = jnp.asarray(grid.log_n[1], dtype=jnp.float64)
    z = jnp.asarray(grid.z[2], dtype=jnp.float64)
    E_obs = grid.E_obs
    tau_grid = grid.tau(log_T, log_n, z, E_obs)
    tau_scratch = compute_tau_from_scratch(log_T=log_T, log_n=log_n, z=z, E_obs=E_obs)
    rel = jnp.abs(tau_scratch - tau_grid) / jnp.maximum(jnp.abs(tau_grid), 1e-30)
    assert float(rel.max()) < 0.05
