"""Tests for Observation and ObservationStatic dataclasses."""

from __future__ import annotations

import jax.numpy as jnp


def test_observation_static_construction() -> None:
    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4, 0.5, 0.6]),
        ene_hi=jnp.array([0.5, 0.6, 0.7]),
        chan_lo=jnp.array([0.4, 0.5]),
        chan_hi=jnp.array([0.5, 0.6]),
        rmf=jnp.zeros((2, 3)),
        arf=jnp.array([100.0, 100.0, 100.0]),
        exposure=jnp.float64(1000.0),
        z=jnp.float64(2.0),
        nh_mw=jnp.float64(1e20),
    )
    assert static.n_chan == 2
    assert static.n_ene == 3
    assert static.ene_avg.shape == (3,)
    assert static.ene_width.shape == (3,)
    assert jnp.allclose(static.ene_avg, jnp.array([0.45, 0.55, 0.65]))
    assert jnp.allclose(static.ene_width, jnp.array([0.1, 0.1, 0.1]))


def test_observation_static_is_pytree() -> None:
    """Must be a JAX PyTree so jax.vmap works over a list of them."""
    import jax

    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4]),
        ene_hi=jnp.array([0.5]),
        chan_lo=jnp.array([0.4]),
        chan_hi=jnp.array([0.5]),
        rmf=jnp.zeros((1, 1)),
        arf=jnp.array([100.0]),
        exposure=jnp.float64(1.0),
        z=jnp.float64(1.0),
        nh_mw=jnp.float64(1e20),
    )
    leaves, _ = jax.tree.flatten(static)
    assert len(leaves) >= 7


def test_observation_static_dtype_promotion() -> None:
    """ene_lo etc. must end up float64 even if user passed float32."""
    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4], dtype=jnp.float32),
        ene_hi=jnp.array([0.5], dtype=jnp.float32),
        chan_lo=jnp.array([0.4]),
        chan_hi=jnp.array([0.5]),
        rmf=jnp.zeros((1, 1)),
        arf=jnp.array([100.0]),
        exposure=jnp.float64(1.0),
        z=jnp.float64(1.0),
        nh_mw=jnp.float64(1e20),
    )
    assert static.ene_lo.dtype == jnp.float64
    assert static.ene_hi.dtype == jnp.float64


def test_observation_holds_counts() -> None:
    from whimstan.data.types import Observation, ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4, 0.5]),
        ene_hi=jnp.array([0.5, 0.6]),
        chan_lo=jnp.array([0.4, 0.5]),
        chan_hi=jnp.array([0.5, 0.6]),
        rmf=jnp.eye(2),
        arf=jnp.array([100.0, 100.0]),
        exposure=jnp.float64(1000.0),
        z=jnp.float64(2.0),
        nh_mw=jnp.float64(1e20),
    )
    obs = Observation(
        name="GRB000000",
        static=static,
        counts=jnp.array([10, 12]),
        bkg_counts=jnp.array([2, 3]),
        bkg_scale=jnp.array([0.05, 0.05]),
        ra_deg=0.0,
        dec_deg=0.0,
    )
    assert obs.name == "GRB000000"
    assert obs.counts.dtype == jnp.int32 or obs.counts.dtype == jnp.int64
    assert obs.bkg_scale.shape == (2,)
