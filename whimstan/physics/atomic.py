"""Per-ion photoionization cross sections (Verner & Yakovlev 1995).

sigma_nl(E) = sigma_0 * F(y),  y = E / E_0,
F(y) = [(y - 1)^2 + y_w^2] * y^{-Q} * (1 + sqrt(y / y_a))^{-P},
Q = 5.5 + l - 0.5 P,  valid above the shell threshold E_th.

Total per-ion sigma = sum over occupied shells. Fit parameters are
committed in _data/vy95_phfit_v1.npz (built by tools/build_vy95_table.py
from VizieR J/A+AS/109/125).

This module also owns the element/ion registry shared by the ionization
solver, the UVB Gamma table, and the grid assembler.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array

ELEMENTS: tuple[tuple[str, int], ...] = (
    ("H", 1),
    ("He", 2),
    ("C", 6),
    ("N", 7),
    ("O", 8),
    ("Ne", 10),
    ("Mg", 12),
    ("Si", 14),
    ("S", 16),
    ("Ar", 18),
    ("Ca", 20),
    ("Fe", 26),
)

#: number of absorbing ions (>=1 electron): stages 0..Z-1 per element.
N_IONS = sum(z for _, z in ELEMENTS)  # 140
#: number of ionization stages incl. bare nucleus: 0..Z per element.
N_STAGES = sum(z + 1 for _, z in ELEMENTS)  # 152

#: start index of each element's block in the 140-layout / 152-layout.
ION_OFFSETS: dict[str, int] = {}
STAGE_OFFSETS: dict[str, int] = {}
_o = _s = 0
for _name, _z in ELEMENTS:
    ION_OFFSETS[_name] = _o
    STAGE_OFFSETS[_name] = _s
    _o += _z
    _s += _z + 1
del _o, _s, _name, _z


def ion_index(element: str, stage: int) -> int:
    """Flat 140-layout index for `element` with `stage` electrons removed."""
    z = dict(ELEMENTS)[element]
    if not 0 <= stage < z:
        raise ValueError(f"{element} stage {stage} has no bound electrons")
    return ION_OFFSETS[element] + stage


_DATA = Path(__file__).parent / "_data" / "vy95_phfit_v1.npz"
with np.load(_DATA) as _f:
    _ETH = jnp.asarray(_f["E_th_eV"])  # (140, 7)
    _E0 = jnp.asarray(_f["E0_eV"])
    _S0 = jnp.asarray(_f["sigma0_Mb"])
    _YA = jnp.asarray(_f["y_a"])
    _P = jnp.asarray(_f["P"])
    _YW = jnp.asarray(_f["y_w"])
    _LQ = jnp.asarray(_f["l_q"])
    _MASK = jnp.asarray(_f["shell_mask"])

# Guard against log(0)/0-division on padded shells.
_E0_SAFE = jnp.where(_MASK, _E0, 1.0)
_YA_SAFE = jnp.where(_MASK, _YA, 1.0)


def sigma_ions(E_keV: Array) -> Array:
    """Cross section per absorbing ion, cm^2.

    Parameters
    ----------
    E_keV : (N_E,) photon energies, keV.

    Returns
    -------
    (N_IONS, N_E) array; zero below each shell threshold.
    """
    E_eV = jnp.asarray(E_keV, dtype=jnp.float64) * 1.0e3
    y = E_eV[None, None, :] / _E0_SAFE[:, :, None]  # (140, 7, NE)
    q = 5.5 + _LQ - 0.5 * _P  # (140, 7)
    f = (
        ((y - 1.0) ** 2 + _YW[:, :, None] ** 2)
        * y ** (-q[:, :, None])
        * (1.0 + jnp.sqrt(y / _YA_SAFE[:, :, None])) ** (-_P[:, :, None])
    )
    sig = _S0[:, :, None] * 1.0e-18 * f  # Mb -> cm^2
    keep = (E_eV[None, None, :] >= _ETH[:, :, None]) & _MASK[:, :, None]
    return jnp.where(keep, sig, 0.0).sum(axis=1)
