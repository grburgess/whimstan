"""Tests for WhimGrid loader and JAX interpolator."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="module")
def tiny_grid_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "whim_grid_tiny.h5"
    assert p.is_file(), f"run tests/fixtures/build_whim_grid_tiny.py first; missing {p}"
    return p


def test_load(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    assert g.tau_table.shape == (5, 4, 6, 64)
    assert g.log_T.shape == (5,)
    assert g.log_n.shape == (4,)
    assert g.z.shape == (6,)
    assert g.E_obs.shape == (64,)
    assert g.metadata["grid_version"] == "synthetic-tiny-v0"


def test_interp_at_grid_corner_exact(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    tau_interp = g.tau(g.log_T[2], g.log_n[1], g.z[3], g.E_obs)
    expected = g.tau_table[2, 1, 3, :]
    assert jnp.allclose(tau_interp, expected, rtol=1e-5)


def test_interp_monotone_in_n(tiny_grid_path: Path) -> None:
    """tau must be monotone increasing in log n_H at fixed (T, z, E)."""
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    n_grid = jnp.linspace(-6.5, -3.5, 20)
    tau_n = jax.vmap(lambda lognh: g.tau(6.0, lognh, 1.0, g.E_obs))(n_grid)
    tau_total = tau_n.sum(axis=1)
    diffs = jnp.diff(tau_total)
    assert jnp.all(diffs >= 0), f"non-monotone: {diffs}"


def test_jit_and_vmap(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)

    @jax.jit
    def fold(log_T, log_n, z):
        return g.tau(log_T, log_n, z, g.E_obs).sum()

    out = fold(jnp.float64(6.0), jnp.float64(-5.0), jnp.float64(2.0))
    assert jnp.isfinite(out)

    log_T = jnp.array([5.5, 6.0, 6.5])
    log_n = jnp.array([-5.0, -5.0, -5.0])
    z = jnp.array([1.0, 2.0, 3.0])
    out_v = jax.vmap(fold)(log_T, log_n, z)
    assert out_v.shape == (3,)


def test_grad_finite(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)

    def loss(p):
        return g.tau(p[0], p[1], p[2], g.E_obs).sum()

    p = jnp.array([6.0, -5.0, 2.0])
    grads = jax.grad(loss)(p)
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.any(grads != 0)


def test_clip_at_grid_edge(tiny_grid_path: Path) -> None:
    """Out-of-range queries clip to the boundary, do not nan."""
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    out = g.tau(jnp.float64(8.0), jnp.float64(-2.0), jnp.float64(10.0), g.E_obs)
    expected = g.tau_table[-1, -1, -1, :]
    assert jnp.allclose(out, expected, rtol=1e-5)
