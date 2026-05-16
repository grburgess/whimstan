"""Generate a tiny synthetic WHIM grid for tests.

Real grid (Plan 2) replaces this with a 50 MB CLOUDY-derived HDF5.
Synthetic tau is a smooth analytic function so monotonicity and
gradient tests are well-defined.
"""

from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    log_T = np.linspace(4.5, 7.5, 5)  # 5 T points
    log_n = np.linspace(-7.0, -3.0, 4)  # 4 n_H points
    z = np.linspace(0.0, 5.0, 6)  # 6 z points
    E_obs = np.logspace(np.log10(0.3), np.log10(10.0), 64)  # 64 E points

    sigma_E = (E_obs / 1.0) ** -2.5
    geom_T = np.exp(-((log_T - 6.0) ** 2) / 0.5)  # (5,)
    base = sigma_E[None, None, None, :] * geom_T[:, None, None, None]
    n_factor = (10.0**log_n)[None, :, None, None]
    z_factor = ((1.0 + z) ** 2)[None, None, :, None]
    tau = base * n_factor * z_factor  # (5, 4, 6, 64)

    out = Path("tests/fixtures/whim_grid_tiny.h5")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("log_T", data=log_T)
        f.create_dataset("log_n_H", data=log_n)
        f.create_dataset("z", data=z)
        f.create_dataset("E_obs_keV", data=E_obs)
        f.create_dataset("tau", data=tau.astype("float32"))
        meta = f.create_group("metadata")
        meta.attrs["grid_version"] = "synthetic-tiny-v0"
        meta.attrs["ionizing_background"] = "synthetic"
        meta.attrs["abundances"] = "synthetic"
        meta.attrs["cosmology"] = "synthetic"
    print(f"Wrote {out}: shape={tau.shape}")


if __name__ == "__main__":
    main()
