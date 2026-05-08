"""Smoke tests: package imports and basic invariants."""

from __future__ import annotations


def test_package_imports() -> None:
    import whimstan

    assert hasattr(whimstan, "__version__")


def test_jax_x64_enforced() -> None:
    """JAX must be in float64 mode globally; spec requires this."""
    import jax

    import whimstan  # noqa: F401  triggers x64 enable

    assert jax.config.read("jax_enable_x64") is True


def test_jax_x64_actually_active() -> None:
    """Sanity: a default-constructed array is float64."""
    import jax.numpy as jnp

    import whimstan  # noqa: F401

    x = jnp.array(1.0)
    assert x.dtype == jnp.float64, f"expected float64, got {x.dtype}"
