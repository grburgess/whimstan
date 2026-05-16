"""From-scratch WHIM optical-depth calculator (cross-validator).

PLAN 1 STATUS: STUB.
This module ships an analytic stub whose functional form *matches the
synthetic tiny grid built by tests/fixtures/build_whim_grid_tiny.py*.
This lets us land the public API and unit-test the validator interface
in Plan 1.

PLAN 2 will replace `_analytic_stub_tau` with a real Saha-Boltzmann +
photoionization computation using the same atomic data the CLOUDY grid
build consumes. The public function signature `compute_tau_from_scratch`
must NOT change between Plan 1 and Plan 2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _analytic_stub_tau(log_T: Array, log_n: Array, z: Array, E_obs: Array) -> Array:
    """Smooth analytic tau; mirrors tests/fixtures/build_whim_grid_tiny.py."""
    sigma_E = (E_obs / 1.0) ** -2.5
    geom_T = jnp.exp(-((log_T - 6.0) ** 2) / 0.5)
    n_factor = 10.0**log_n
    z_factor = (1.0 + z) ** 2
    return sigma_E * geom_T * n_factor * z_factor


def compute_tau_from_scratch(
    *,
    log_T: Array | float,
    log_n: Array | float,
    z: Array | float,
    E_obs: Array,
) -> Array:
    """Compute observer-frame tau(E_obs) from-scratch in pure JAX.

    Returns
    -------
    (N_E,) array of optical depths.

    Notes
    -----
    Plan 1 implementation is an analytic stub matching the synthetic
    tiny grid for interface tests. Plan 2 replaces with full physics.
    """
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    log_n = jnp.asarray(log_n, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    E_obs = jnp.asarray(E_obs, dtype=jnp.float64)
    return _analytic_stub_tau(log_T, log_n, z, E_obs)
