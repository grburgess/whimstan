"""Dataclasses for one Swift-XRT observation.

`ObservationStatic` is the JAX-PyTree-registered subset consumed by the
forward model. `Observation` wraps it with raw counts and metadata used
for diagnostics, PPCs, and serialization.

Per the jaxspec spike, `Observation.bkg_scale` is a per-channel array
(alpha = effective_source_exposure / effective_background_exposure), the
jaxspec `folded_backratio`. This is used directly by the W-stat profile
likelihood. We do NOT carry a scalar bkg_exposure because real OGIP
files have channel-dependent BACKSCAL/AREASCAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def _as_f64(x: Any) -> Array:
    """Coerce input to JAX float64 array."""
    return jnp.asarray(x, dtype=jnp.float64)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationStatic:
    """Per-GRB precomputed inputs for the forward model.

    All fields are JAX arrays of dtype float64. Registered as a PyTree
    so it can be `vmap`'d over.

    Attributes
    ----------
    ene_lo, ene_hi : (N_ene,)
        Source-spectrum energy bin edges in keV (observer frame).
    chan_lo, chan_hi : (N_chan,)
        Detector channel bin edges in keV.
    rmf : (N_chan, N_ene)
        Response matrix; rmf @ source_flux gives expected channel rate.
    arf : (N_ene,)
        Effective area in cm^2.
    exposure : scalar
        Source exposure in seconds.
    z : scalar
        Source redshift.
    nh_mw : scalar
        Milky-Way column density in cm^-2 (fixed; not a parameter).
    """

    ene_lo: Array
    ene_hi: Array
    chan_lo: Array
    chan_hi: Array
    rmf: Array
    arf: Array
    exposure: Array
    z: Array
    nh_mw: Array

    def __post_init__(self) -> None:
        for name in (
            "ene_lo",
            "ene_hi",
            "chan_lo",
            "chan_hi",
            "rmf",
            "arf",
            "exposure",
            "z",
            "nh_mw",
        ):
            object.__setattr__(self, name, _as_f64(getattr(self, name)))

    @property
    def n_chan(self) -> int:
        return int(self.chan_lo.shape[0])

    @property
    def n_ene(self) -> int:
        return int(self.ene_lo.shape[0])

    @property
    def ene_avg(self) -> Array:
        return 0.5 * (self.ene_lo + self.ene_hi)

    @property
    def ene_width(self) -> Array:
        return self.ene_hi - self.ene_lo


@dataclass(frozen=True)
class Observation:
    """One Swift-XRT GRB observation with raw counts + metadata.

    Not a JAX PyTree; carries Python-side fields. Use `static` to access
    the JAX-side state for the forward model.
    """

    name: str
    static: ObservationStatic
    counts: Array  # (N_chan,) int
    bkg_counts: Array  # (N_chan,) int - folded_background_unscaled
    bkg_scale: Array  # (N_chan,) float - folded_backratio
    ra_deg: float
    dec_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", jnp.asarray(self.counts))
        object.__setattr__(self, "bkg_counts", jnp.asarray(self.bkg_counts))
        object.__setattr__(self, "bkg_scale", _as_f64(self.bkg_scale))
