"""Wilms+ 2000 TbAbs photoelectric ISM cross section, JAX-native.

Loaded from a precomputed npz table shipped under ``physics/_data/``. The
table is generated offline by ``tools/build_tbabs_table.py`` (Plan-1
Task 18 used path B -- ``astromodels.functions.TbAbs`` -- to evaluate
transmission ``t(E)`` at ``NH = 1`` in units of ``1e22 cm^-2`` and
inverted via ``sigma(E) = -ln(t(E)) / 1e22``). This module exposes a
single function ``tbabs_sigma(E)`` that does log-log linear interpolation
in JAX.

Notes
-----
* ``tbabs_sigma(E)`` returns sigma per H atom (cm^2). To get optical
  depth multiply by NH: ``tau = NH * tbabs_sigma(E)``.
* For E outside the table, we clip to the nearest grid value (no
  extrapolation); a logged warning would be added in higher layers.
"""

from __future__ import annotations

from importlib.resources import files

import jax.numpy as jnp
import numpy as np
from jax import Array

_NPZ_PATH = files("whimstan.physics._data").joinpath("tbabs_sigma_v1.npz")
with _NPZ_PATH.open("rb") as _f:
    _NPZ = np.load(_f)
    TBABS_E_KEV: Array = jnp.asarray(_NPZ["E_keV"], dtype=jnp.float64)
    TBABS_SIGMA_CM2: Array = jnp.asarray(_NPZ["sigma_cm2"], dtype=jnp.float64)
del _NPZ, _NPZ_PATH

_LOG_E: Array = jnp.log(TBABS_E_KEV)
_LOG_S: Array = jnp.log(TBABS_SIGMA_CM2)


def tbabs_sigma(E_keV: Array | float) -> Array:
    """Photoelectric cross section per H atom (cm^2) at energies E_keV.

    Log-log linear interpolation; clips at table edges.
    """
    E = jnp.asarray(E_keV, dtype=jnp.float64)
    logE = jnp.log(E)
    return jnp.exp(jnp.interp(logE, _LOG_E, _LOG_S))
