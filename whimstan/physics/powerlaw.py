"""Photon power-law normalized to integrated energy flux.

The photon spectrum is

    N(E) = K_norm * (E / E_piv) ** gamma                [photons / cm^2 / s / keV]

with K_norm chosen so that the integrated *energy* flux

    sum_E E * N(E) * dE * keV2erg

over [e_lo, e_hi] equals K = 10 ** log_K   [erg / cm^2 / s].

Spec §3.1 reasoning: this parameterization decorrelates K from gamma in
the posterior compared to the standard photon-flux normalization.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

KEV_TO_ERG: float = 1.602_176_634e-9  # CODATA 2019


def powerlaw_eflux(
    E: Array,
    log_K: Array | float,
    gamma: Array | float,
    e_lo: float,
    e_hi: float,
    e_piv: float,
) -> Array:
    """Photon spectrum N(E) (photons/cm^2/s/keV) for an eflux-normalized PL.

    Parameters
    ----------
    E : (N,)
        Energies in keV at which to evaluate N(E).
    log_K
        log10 of the integrated energy flux in [e_lo, e_hi], in erg/cm^2/s.
    gamma
        Photon-index slope (typical GRB index ≈ -2).
    e_lo, e_hi, e_piv
        Band edges (keV) for the energy-flux normalization and pivot.

    Returns
    -------
    (N,) array of photon flux density in photons/cm^2/s/keV at each E.
    """
    E = jnp.asarray(E, dtype=jnp.float64)
    log_K = jnp.asarray(log_K, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)

    # Analytic integral of E * (E / e_piv)^gamma dE from e_lo to e_hi.
    # Let p = gamma + 1. Integrand = e_piv^(-gamma) * E^(gamma+1).
    p = gamma + 1.0
    p_eq_minus_one = jnp.isclose(p, -1.0)
    safe_p = jnp.where(p_eq_minus_one, 0.0, p)
    log_term = jnp.log(e_hi / e_lo)
    pow_term = (e_hi ** (safe_p + 1) - e_lo ** (safe_p + 1)) / (safe_p + 1)
    integral_keV = jnp.where(
        p_eq_minus_one,
        e_piv**2 * log_term,
        e_piv ** (-gamma) * pow_term,
    )
    K_norm = (10.0**log_K) / (KEV_TO_ERG * integral_keV)
    return K_norm * (E / e_piv) ** gamma
