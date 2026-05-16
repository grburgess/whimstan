"""Pure-JAX physics primitives for the forward model."""

from __future__ import annotations

from whimstan.physics.cosmology import (
    FlatLambdaCDM,
    Planck18,
    comoving_distance_Mpc,
    dimensionless_hubble,
)
from whimstan.physics.powerlaw import KEV_TO_ERG, powerlaw_eflux
from whimstan.physics.tbabs import TBABS_E_KEV, TBABS_SIGMA_CM2, tbabs_sigma
from whimstan.physics.whim import WhimGrid
from whimstan.physics.whim_validator import compute_tau_from_scratch

__all__ = [
    "KEV_TO_ERG",
    "TBABS_E_KEV",
    "TBABS_SIGMA_CM2",
    "FlatLambdaCDM",
    "Planck18",
    "WhimGrid",
    "comoving_distance_Mpc",
    "compute_tau_from_scratch",
    "dimensionless_hubble",
    "powerlaw_eflux",
    "tbabs_sigma",
]
