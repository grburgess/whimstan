"""Dataset = ordered collection of Observations.

Provides stacking utilities for broadcasting per-GRB static fields into
the leading axis of a single PyTree, ready for vmap.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from whimstan.data.types import Observation, ObservationStatic


@dataclass(frozen=True)
class Dataset:
    """Ordered list of Observations all sharing N_chan and N_ene."""

    observations: tuple[Observation, ...]

    def __init__(self, observations: Iterable[Observation]) -> None:
        obs_tuple = tuple(observations)
        if not obs_tuple:
            raise ValueError("Dataset requires at least one Observation")
        n_chan = obs_tuple[0].static.n_chan
        n_ene = obs_tuple[0].static.n_ene
        for o in obs_tuple[1:]:
            if o.static.n_chan != n_chan or o.static.n_ene != n_ene:
                raise ValueError(
                    f"observation {o.name!r} has shape "
                    f"({o.static.n_chan}, {o.static.n_ene}) but expected "
                    f"({n_chan}, {n_ene}); v1.0 requires uniform grids"
                )
        object.__setattr__(self, "observations", obs_tuple)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Observation:
        return self.observations[idx]

    @property
    def n_grbs(self) -> int:
        return len(self.observations)

    @property
    def n_chan(self) -> int:
        return self.observations[0].static.n_chan

    @property
    def n_ene(self) -> int:
        return self.observations[0].static.n_ene

    def stack_static(self) -> ObservationStatic:
        """Stack all observations' static state along axis 0.

        Returns a single ObservationStatic whose fields have a leading
        N_grbs axis. Convenient for `vmap`.
        """
        statics: Sequence[ObservationStatic] = [o.static for o in self.observations]
        return jax.tree.map(  # type: ignore[no-any-return]
            lambda *xs: jnp.stack(xs, axis=0),
            *statics,
        )

    def total_counts(self) -> jax.Array:
        """Per-GRB total source counts; used by reparam centering threshold."""
        return jnp.stack([o.counts.sum() for o in self.observations])
