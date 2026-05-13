"""Build the TbAbs cross-section lookup table shipped with whimstan.

Produces ``whimstan/physics/_data/tbabs_sigma_v1.npz`` containing:

* ``E_keV``    : log-spaced energy grid 0.1-100 keV, length 4096.
* ``sigma_cm2``: photoelectric cross section per H atom (cm^2).

Generation path used (Plan-1 Task 18): **B** -- ``astromodels.functions.TbAbs``.

``astromodels`` ships the Wilms+ 2000 TbAbs transmission as a tabulated
function with solar abundances. We evaluate transmission ``t(E)`` at
``NH = 1`` (units of 1e22 cm^-2, ``z = 0``) and invert via
``sigma(E) = -ln(t(E)) / 1e22`` to recover the per-atom cross section.

Run::

    uv tool run --with astromodels --with numpy python tools/build_tbabs_table.py

This script is intentionally standalone; ``astromodels`` is a heavy dep
that is not part of the runtime requirements.
"""

from __future__ import annotations

import pathlib

import numpy as np
from astromodels.functions import TbAbs

PLACEHOLDER = False  # Path B: real Wilms+ 2000 TbAbs via astromodels.

OUT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "whimstan/physics/_data/tbabs_sigma_v1.npz"
)


def main() -> None:
    tb = TbAbs()
    E = np.logspace(np.log10(0.1), np.log10(100.0), 4096)
    transmission = np.asarray(tb.evaluate(E, 1.0, 0.0), dtype=np.float64)
    sigma_per_1e22 = -np.log(np.clip(transmission, 1e-300, 1.0))
    sigma = sigma_per_1e22 * 1e-22

    assert (sigma > 0).all(), "sigma must be positive everywhere"
    assert sigma.shape == E.shape

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, E_keV=E, sigma_cm2=sigma)
    print(f"wrote {OUT}: sigma range [{sigma.min():.3e}, {sigma.max():.3e}] cm^2/H")


if __name__ == "__main__":
    main()
