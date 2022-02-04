from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

from . import get_path_of_data_file


class AbsoriCalculations:
    def __init__(self) -> None:
        """
        Opens and holds the precomputed Absori information
        """
        self._data_file: Path = get_path_of_data_file("absori.h5")

        self._absori_elements = (
            "H",
            "He",
            "C",
            "N",
            "O",
            "Ne",
            "Mg",
            "Si",
            "S",
            "Fe",
        )

        with h5py.File(self._data_file, "r") as f:

            self._ion = f["ion"][()]
            self._sigma = f["sigma"][()]
            self._energy = f["energy"][()]
            self._atomic_number = f["atomicnumber"][()]

    def get_spec(self, gamma=2) -> np.ndarray:
        assert gamma == 2, "Only for gamma=2 at the moment"
        return np.load(get_path_of_data_file("spec_gamma2.npy"))

    @property
    def ion(self) -> np.ndarray:
        return self._ion

    @property
    def sigma(self) -> np.ndarray:
        return self._sigma

    @property
    def energy(self) -> np.ndarray:
        return self._energy

    @property
    def atomic_number(self) -> np.ndarray:
        return self._atomic_number

    def get_abundance(self, name: str = "angr") -> np.ndarray:

        out = np.empty(len(self._absori_elements))

        with h5py.File(self._data_file, "r") as f:

            abund_grp: h5py.Group = f["abundances"]

            name_grp: h5py.Group = abund_grp[name]

            for i, element in enumerate(self._absori_elements):

                out[i] = name_grp.attrs[element]

        return out


@dataclass
class CosmoConstants:
    omegam: float = 0.307
    omegal: float = 0.693
    h0: float = 67.7
    c: float = 2.99792458e5
    cmpermpc: float = 3.08568e24


def interpolate_sigma(ekeV, energy_base, sigma_base) -> np.ndarray:
    e = 1000 * ekeV
    res = np.zeros((e.shape[0], e.shape[1], 26, 10))
    mask1 = e > energy_base[-1]
    mask2 = e < energy_base[0]
    mask3 = (~mask1) * (~mask2)

    sigma_interp = interp1d(energy_base, sigma_base, axis=0)
    res[mask3] = sigma_interp(e[mask3])

    res[mask1] = np.expand_dims(sigma_base[720], axis=0)
    res[mask1] *= np.expand_dims(
        np.power((e[mask1] / energy_base[-1]), -3.0), axis=(1, 2)
    )

    res[mask2] = np.expand_dims(sigma_base[0], axis=0)
    return res


def sum_sigma_interp_precalc(
    z, x, energy_base, sigma_base, zshell_thickness: float = 0.02
) -> np.ndarray:
    nz = int(z / zshell_thickness)
    zsam = z / nz
    zz = zsam * 0.5

    # all the different redshifted energies in the
    # z shells
    energy_z = np.zeros((len(x), nz))
    # weight factors from z integral and constants
    zf = np.zeros(nz)

    # loop through shells
    for i in range(nz):
        z1 = zz + 1
        energy_z[:, i] = z1 * x
        zf[i] = z1 ** 2 / np.sqrt(
            CosmoConstants.omegam * (z1 ** 3) + CosmoConstants.omegal
        )
        zz += zsam
    zf *= (
        zsam
        * CosmoConstants.c
        * CosmoConstants.cmpermpc
        / CosmoConstants.h0
        * 6.6e-5
        * 1e-22
    )
    sigma_inter = interpolate_sigma(energy_z, energy_base, sigma_base)
    sigma_inter = np.swapaxes(sigma_inter, 0, 1)
    sigma_inter = np.swapaxes(sigma_inter, 2, 3)

    return np.sum(sigma_inter.T * zf, axis=-1).T
