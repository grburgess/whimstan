"""whimstan — Hierarchical Bayesian WHIM search in GRB afterglows."""

from __future__ import annotations

# Enforce float64 globally before any other JAX import side-effects.
# Spec §7.1: "JAX_ENABLE_X64=True enforced at module import; checked on startup".
import jax as _jax

_jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call,unused-ignore]
del _jax

from whimstan import data, physics, utils  # noqa: E402
from whimstan.__about__ import __version__  # noqa: E402

__all__ = ["__version__", "data", "physics", "utils"]
