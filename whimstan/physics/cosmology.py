"""FlatLambdaCDM in pure JAX.

Spec §5.3: Planck18 cosmology drives the WHIM line-of-sight integral.
Used at *grid-build time* only — inference-time JAX code never sees this
because the cosmological integral is already baked into the grid. We
keep this module so the JAX validator (Task 20) and the simulation
pipeline can compute distances differentiably without astropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

# Speed of light, km/s.
_C_KM_S = 299_792.458


@dataclass(frozen=True, slots=True)
class FlatLambdaCDM:
    """Flat-LambdaCDM cosmology; matter and dark-energy fractions sum to 1."""

    H0_km_s_Mpc: float
    Omega_m: float
    Omega_L: float

    @property
    def hubble_distance_Mpc(self) -> float:
        return _C_KM_S / self.H0_km_s_Mpc


# Planck 2018 TT,TE,EE+lowE+lensing+BAO best-fit (rounded).
Planck18 = FlatLambdaCDM(H0_km_s_Mpc=67.4, Omega_m=0.315, Omega_L=0.685)


def dimensionless_hubble(z: Array | float, cosmo: FlatLambdaCDM) -> Array:
    """E(z) = sqrt(Omega_m (1+z)^3 + Omega_L). Flat universe."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return jnp.sqrt(cosmo.Omega_m * (1.0 + z) ** 3 + cosmo.Omega_L)


def comoving_distance_Mpc(
    z: Array | float, cosmo: FlatLambdaCDM, n_steps: int = 4096
) -> Array:
    """Comoving distance D_C(z) in Mpc.

    Uses fixed-step trapezoidal integration of c/H(z') from 0 to z. Step
    count is configurable so the validator can tighten convergence;
    default 4096 is well under 0.1% error vs astropy on z<5.
    """
    z = jnp.asarray(z, dtype=jnp.float64)

    def integrand(zp: Array) -> Array:
        return 1.0 / dimensionless_hubble(zp, cosmo)

    def D_one(zi: Array) -> Array:
        zp = jnp.linspace(0.0, zi, n_steps)
        y = integrand(zp)
        return cosmo.hubble_distance_Mpc * jnp.trapezoid(y, zp)

    if z.ndim == 0:
        return D_one(z)
    return jax.vmap(D_one)(z)
