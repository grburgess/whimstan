"""jaxspec-backed loader for OGIP-format X-ray spectral files.

Thin wrapper: this is the only module that imports jaxspec. If jaxspec
ever needs to be replaced, only this file (plus tests) changes.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from whimstan.data.types import Observation, ObservationStatic


def load_swift_xrt_observation(
    *,
    name: str,
    pha_path: str | Path,
    rmf_path: str | Path,
    arf_path: str | Path,
    bkg_path: str | Path,
    z: float,
    nh_mw: float,
    ra_deg: float,
    dec_deg: float,
    e_lo: float = 0.4,
    e_hi: float = 10.0,
) -> Observation:
    """Load one Swift-XRT observation from PHA/RMF/ARF/BKG files.

    Parameters
    ----------
    name
        Identifier (e.g. GRB name).
    pha_path, rmf_path, arf_path, bkg_path
        Filesystem paths to OGIP files.
    z, nh_mw
        Source redshift and Milky-Way column density (cm^-2). Both fixed.
    ra_deg, dec_deg
        Sky position for downstream MW-NH lookups.
    e_lo, e_hi
        Channel-energy band kept; channels outside [e_lo, e_hi] are dropped.

    Returns
    -------
    Observation
        Wrapped Observation with float64 JAX arrays inside ObservationStatic.

    Raises
    ------
    FileNotFoundError
        If any of the input paths does not exist.
    """
    for p in (pha_path, rmf_path, arf_path, bkg_path):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    # jaxspec 0.2.0 transitively emits third-party DeprecationWarnings during
    # import + load (numpyro accessing `jax.core.call_p`, `sparse.COO` coords
    # dtype warning). They are out of our control, so suppress DeprecationWarning
    # for the duration of the call to keep pytest's `filterwarnings=error` happy.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from jaxspec.data import ObsConfiguration

        cfg = ObsConfiguration.from_pha_file(
            pha_path=str(pha_path),
            rmf_path=str(rmf_path),
            arf_path=str(arf_path),
            bkg_path=str(bkg_path),
            low_energy=e_lo,
            high_energy=e_hi,
        )

    # jaxspec 0.2.0 attribute mapping (see Task 5 spike findings).
    # in_energies and out_energies are shape (2, N): [0,:]=lo, [1,:]=hi.
    counts = jnp.asarray(cfg.folded_counts.values)
    bkg_counts = jnp.asarray(cfg.folded_background_unscaled.values)
    bkg_scale = jnp.asarray(cfg.folded_backratio.values)
    exposure = jnp.asarray(cfg.exposure.values)

    in_lo = jnp.asarray(cfg.in_energies[0, :])
    in_hi = jnp.asarray(cfg.in_energies[1, :])
    out_lo = jnp.asarray(cfg.out_energies[0, :])
    out_hi = jnp.asarray(cfg.out_energies[1, :])

    # transfer_matrix is backed by a sparse.COO that refuses implicit densify.
    rmf_raw = cfg.transfer_matrix.data
    rmf_dense = rmf_raw.todense() if hasattr(rmf_raw, "todense") else rmf_raw
    rmf = jnp.asarray(rmf_dense)
    arf = jnp.asarray(cfg.area.values)

    static = ObservationStatic(
        ene_lo=in_lo,
        ene_hi=in_hi,
        chan_lo=out_lo,
        chan_hi=out_hi,
        rmf=rmf,
        arf=arf,
        exposure=exposure,
        z=jnp.float64(z),
        nh_mw=jnp.float64(nh_mw),
    )
    return Observation(
        name=name,
        static=static,
        counts=counts,
        bkg_counts=bkg_counts,
        bkg_scale=bkg_scale,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
