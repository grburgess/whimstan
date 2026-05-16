"""WHIM opacity grid loader + JAX trilinear interpolator.

The grid is a 4-D array tau[i_T, i_n, i_z, i_E] of optical depths in
the observer frame. At inference time we trilinearly interpolate over
(log T, log n_H, z) at every requested (T, n_H, z), producing a vector
over the grid's E_obs axis. The caller (forward model) then linearly
interpolates that vector onto the per-GRB observer-frame energy grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
from jax import Array


def _searchsorted_clipped(grid: Array, x: Array) -> tuple[Array, Array]:
    """Return (i, w) such that linear interp uses grid[i] and grid[i+1]
    with weight w in [0, 1]. Clips x to grid range (no extrapolation).
    """
    n = grid.shape[0]
    i = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, n - 2)
    x_clipped = jnp.clip(x, grid[0], grid[-1])
    w = (x_clipped - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


@dataclass(frozen=True, slots=True)
class WhimGrid:
    """Tabulated WHIM optical depth tau(log_T, log_n_H, z, E_obs)."""

    log_T: Array  # (N_T,)
    log_n: Array  # (N_n,)
    z: Array  # (N_z,)
    E_obs: Array  # (N_E,) keV
    tau_table: Array  # (N_T, N_n, N_z, N_E) float64
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> WhimGrid:
        with h5py.File(path, "r") as f:
            log_T = jnp.asarray(f["log_T"][...], dtype=jnp.float64)
            log_n = jnp.asarray(f["log_n_H"][...], dtype=jnp.float64)
            z = jnp.asarray(f["z"][...], dtype=jnp.float64)
            E_obs = jnp.asarray(f["E_obs_keV"][...], dtype=jnp.float64)
            tau = jnp.asarray(f["tau"][...], dtype=jnp.float64)
            meta_grp = f.get("metadata", None)
            metadata: dict[str, Any] = {}
            if meta_grp is not None:
                for k, v in meta_grp.attrs.items():
                    metadata[k] = v if not isinstance(v, bytes) else v.decode()
        return cls(
            log_T=log_T,
            log_n=log_n,
            z=z,
            E_obs=E_obs,
            tau_table=tau,
            metadata=metadata,
        )

    def tau(
        self,
        log_T: Array | float,
        log_n: Array | float,
        z: Array | float,
        E_query: Array,
    ) -> Array:
        """Interpolate tau at (log_T, log_n, z) and project to E_query.

        Parameters
        ----------
        log_T, log_n, z : scalars
            Query point in (log T, log n_H, z) space; clipped to grid.
        E_query : (N_E_q,)
            Energies (keV, observer frame) at which to return tau.

        Returns
        -------
        tau_at_E : (N_E_q,)
            Optical depth at each requested energy.
        """
        log_T = jnp.asarray(log_T, dtype=jnp.float64)
        log_n = jnp.asarray(log_n, dtype=jnp.float64)
        z = jnp.asarray(z, dtype=jnp.float64)

        iT, wT = _searchsorted_clipped(self.log_T, log_T)
        iN, wN = _searchsorted_clipped(self.log_n, log_n)
        iZ, wZ = _searchsorted_clipped(self.z, z)

        c000 = self.tau_table[iT, iN, iZ, :]
        c001 = self.tau_table[iT, iN, iZ + 1, :]
        c010 = self.tau_table[iT, iN + 1, iZ, :]
        c011 = self.tau_table[iT, iN + 1, iZ + 1, :]
        c100 = self.tau_table[iT + 1, iN, iZ, :]
        c101 = self.tau_table[iT + 1, iN, iZ + 1, :]
        c110 = self.tau_table[iT + 1, iN + 1, iZ, :]
        c111 = self.tau_table[iT + 1, iN + 1, iZ + 1, :]

        c00 = c000 * (1 - wZ) + c001 * wZ
        c01 = c010 * (1 - wZ) + c011 * wZ
        c10 = c100 * (1 - wZ) + c101 * wZ
        c11 = c110 * (1 - wZ) + c111 * wZ

        c0 = c00 * (1 - wN) + c01 * wN
        c1 = c10 * (1 - wN) + c11 * wN

        tau_at_grid_E = c0 * (1 - wT) + c1 * wT  # (N_E_grid,)

        logE_grid = jnp.log(self.E_obs)
        logE_q = jnp.log(jnp.asarray(E_query, dtype=jnp.float64))
        return jnp.interp(logE_q, logE_grid, tau_at_grid_E)


# Register WhimGrid as a PyTree so it can be carried inside jit'd
# functions without being differentiated through the metadata dict.
# AuxData must be hashable, so metadata is flattened to a tuple of items.
def _flatten(g: WhimGrid) -> tuple[tuple[Array, ...], tuple[tuple[str, Any], ...]]:
    children = (g.log_T, g.log_n, g.z, g.E_obs, g.tau_table)
    aux = tuple(sorted(g.metadata.items()))
    return children, aux


def _unflatten(
    aux: tuple[tuple[str, Any], ...], children: tuple[Array, ...]
) -> WhimGrid:
    log_T, log_n, z, E_obs, tau_table = children
    return WhimGrid(
        log_T=log_T,
        log_n=log_n,
        z=z,
        E_obs=E_obs,
        tau_table=tau_table,
        metadata=dict(aux),
    )


jax.tree_util.register_pytree_node(WhimGrid, _flatten, _unflatten)
