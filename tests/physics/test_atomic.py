"""Tests for VY95 photoionization cross sections."""

from __future__ import annotations

import jax.numpy as jnp


def test_registry_shapes() -> None:
    from whimstan.physics.atomic import ELEMENTS, N_IONS, ion_index  # noqa: F401

    assert N_IONS == 140
    assert ion_index("H", 0) == 0
    assert ion_index("He", 1) == 2
    assert ion_index("Fe", 25) == 139


def test_hydrogen_threshold_value() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    E = jnp.array([0.0137])  # keV, just above 13.6 eV
    sig = sigma_ions(E)
    # sigma(H I, threshold) ~ 6.3e-18 cm^2
    assert 5.0e-18 < float(sig[ion_index("H", 0), 0]) < 8.0e-18


def test_hydrogen_highE_slope() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    E = jnp.array([0.05, 0.1])  # keV
    sig = sigma_ions(E)[ion_index("H", 0)]
    ratio = float(sig[1] / sig[0])
    assert abs(ratio - 2.0**-3) / 2.0**-3 < 0.25  # ~E^-3 far above threshold


def test_threshold_zero_below_edge() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    # O VII (He-like oxygen, stage 6): K edge at 739 eV.
    E = jnp.array([0.5, 0.80])
    sig = sigma_ions(E)[ion_index("O", 6)]
    assert float(sig[0]) == 0.0
    assert float(sig[1]) > 0.0


def test_all_finite_nonnegative() -> None:
    from whimstan.physics.atomic import sigma_ions

    E = jnp.logspace(jnp.log10(0.3), jnp.log10(60.0), 200)
    sig = sigma_ions(E)
    assert sig.shape == (140, 200)
    assert bool(jnp.all(jnp.isfinite(sig)))
    assert bool(jnp.all(sig >= 0.0))
