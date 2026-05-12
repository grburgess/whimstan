"""Tests for energy-flux-normalized power law."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_eflux_integrates_to_K() -> None:
    """Spec §3.1: K_i is the integrated energy flux from E_lo to E_hi."""
    from whimstan.physics.powerlaw import powerlaw_eflux

    E = jnp.linspace(0.4, 10.0, 5000)
    dE = jnp.diff(E)
    Em = 0.5 * (E[:-1] + E[1:])

    log_K = jnp.log10(1.0e-10)  # 1e-10 erg/cm^2/s in 0.4-10 keV
    gamma = -2.0
    photons = powerlaw_eflux(Em, log_K, gamma, e_lo=0.4, e_hi=10.0, e_piv=1.0)
    KEV2ERG = 1.602176634e-9
    eflux_recovered = (photons * Em * dE).sum() * KEV2ERG
    assert pytest.approx(eflux_recovered, rel=2e-3) == 10.0 ** float(log_K)


def test_gamma_2_flat_in_F_E() -> None:
    """gamma = -2 means E^2 N(E) is flat (in keV space)."""
    from whimstan.physics.powerlaw import powerlaw_eflux

    E = jnp.array([0.5, 1.0, 2.0, 5.0])
    photons = powerlaw_eflux(E, log_K=-10.0, gamma=-2.0, e_lo=0.4, e_hi=10.0, e_piv=1.0)
    F_E_squared = E**2 * photons
    rel_spread = (F_E_squared.max() - F_E_squared.min()) / F_E_squared.mean()
    assert float(rel_spread) < 1e-12


def test_jit_and_grad() -> None:
    from whimstan.physics.powerlaw import powerlaw_eflux

    fj = jax.jit(powerlaw_eflux)
    E = jnp.array([1.0, 2.0])
    out = fj(E, -10.0, -2.0, 0.4, 10.0, 1.0)
    assert out.shape == (2,)

    g_log_K = jax.grad(lambda lk: powerlaw_eflux(E, lk, -2.0, 0.4, 10.0, 1.0).sum())(
        -10.0
    )
    assert jnp.isfinite(g_log_K)
    g_gamma = jax.grad(lambda g: powerlaw_eflux(E, -10.0, g, 0.4, 10.0, 1.0).sum())(
        -2.0
    )
    assert jnp.isfinite(g_gamma)
