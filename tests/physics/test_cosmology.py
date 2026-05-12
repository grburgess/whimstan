"""Tests for whimstan.physics.cosmology — JAX FlatLambdaCDM.

Reference values cross-checked against astropy.cosmology Planck18.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def test_planck18_constants() -> None:
    from whimstan.physics.cosmology import Planck18

    assert pytest.approx(Planck18.H0_km_s_Mpc, rel=1e-6) == 67.4
    assert pytest.approx(Planck18.Omega_m, rel=1e-6) == 0.315
    assert pytest.approx(Planck18.Omega_L, rel=1e-6) == 0.685


def test_E_at_z0_is_one() -> None:
    from whimstan.physics.cosmology import Planck18, dimensionless_hubble

    assert float(dimensionless_hubble(0.0, Planck18)) == pytest.approx(1.0)


def test_comoving_distance_matches_astropy() -> None:
    """JAX implementation must match astropy FlatLambdaCDM.comoving_distance to <0.1%.

    Compared against astropy's FlatLambdaCDM constructed with our (rounded)
    Planck18 constants. We avoid astropy.cosmology.Planck18 directly because
    it uses full-precision constants (H0=67.66, Om0=0.30966) plus radiation,
    which differ from our rounded values at the ~0.4% level — that would
    swamp the integration error we actually want to test.
    """
    from astropy.cosmology import FlatLambdaCDM as Astropy_FlatLambdaCDM

    from whimstan.physics.cosmology import Planck18, comoving_distance_Mpc

    ap_cosmo = Astropy_FlatLambdaCDM(
        H0=Planck18.H0_km_s_Mpc, Om0=Planck18.Omega_m, Tcmb0=0.0
    )
    z_grid = jnp.linspace(0.01, 5.0, 50)
    jax_d = np.asarray(comoving_distance_Mpc(z_grid, Planck18, n_steps=4096))
    ap_d = np.asarray(ap_cosmo.comoving_distance(np.asarray(z_grid)).value)
    rel_err = np.abs(jax_d - ap_d) / ap_d
    assert rel_err.max() < 1e-3, f"max rel err {rel_err.max():.3e}"


def test_jit_safe() -> None:
    import jax

    from whimstan.physics.cosmology import Planck18, comoving_distance_Mpc

    f = jax.jit(lambda z: comoving_distance_Mpc(z, Planck18))
    out = f(jnp.array([0.5, 1.0, 2.0]))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))
