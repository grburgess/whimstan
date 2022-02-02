import numpy as np
from astropy.io import fits

from pathlib import Path
import h5py

absori_elements = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Fe"]

names = ["feld", "angr", "aneb","grsa","wilm","lodd","aspl"]



with h5py.File("absori.h5","w") as h5:


    abund_grp = h5.create_group("abundances")

    for name in names:

        name_grp = abund_grp.create_group(name)



        with Path("abundances.dat").open("r") as f:
            rows = f.readlines()
            ele = np.array(rows[0].split(" "), dtype=str)
            ele = ele[ele != ""][1:]
            # get rid of \n at the end
            ele[-1] = ele[-1][:2]
            vals = np.zeros((7, len(ele)))
            keys = []
            for i, row in enumerate(rows[1:8]):
                l = np.array(row.split(" "), dtype=str)
                l = l[l != ""]
                # get rid of \n at the end
                if l[-1][-2:] == "\n":
                    l[-1] = l[-1][:2]
                if l[-1] == "\n":
                    l = l[:-1]
                vals[i] = np.array(l[1:], dtype=float)
                keys.append(l[0][:-1])
            keys = np.array(keys)
        #vals_all = np.zeros(len(absori_elements))

        for i, element in enumerate(absori_elements):
            assert (
                element in ele
            ), f"{element} not a valid element. Valid elements: {ele}"



            idx = np.argwhere(ele == element)[0, 0]

            assert name in keys, f"{name} not a valid name. Valid names: {keys}"

            idy = np.argwhere(keys == name)[0, 0]

            name_grp.attrs[element] = vals[idy, idx]

#            vals_all[i] = vals[idy, idx]

#        return vals_all


    ion = np.zeros((10, 26, 10))
    sigma = np.zeros((10, 26, 721))
    atomicnumber = np.empty(10, dtype=int)

    with fits.open("mansig.fits") as f:
        znumber = f["SIGMAS"].data["Z"]
        ionnumber = f["SIGMAS"].data["ION"]
        sigmadata = f["SIGMAS"].data["SIGMA"]
        iondata = f["SIGMAS"].data["IONDATA"]

        energy = f["ENERGIES"].data["ENERGY"]

    currentZ = -1
    iZ = -1
    iIon = -1
    for i in range(len(znumber)):
        if znumber[i] != currentZ:
            iZ += 1
            atomicnumber[iZ] = znumber[i]
            currentZ = znumber[i]
            iIon = -1
        iIon += 1
        for k in range(10):
            ion[iZ, iIon, k] = iondata[i][k]

        # change units of coef

        ion[iZ][iIon][1] *= 1.0e10
        ion[iZ][iIon][3] *= 1.0e04
        ion[iZ][iIon][4] *= 1.0e-04
        ion[iZ][iIon][6] *= 1.0e-04

        for k in range(721):
            sigma[iZ][iIon][k] = sigmadata[i][k] / 6.6e-27

    elementname = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Fe"]

    ion = ion
    sigma = sigma
    atomicnumber = atomicnumber
    energy = energy

    h5.create_dataset("ion", data = ion, compression="gzip")
    h5.create_dataset("sigma", data = sigma, compression="gzip")
    h5.create_dataset("atomicnumber", data = atomicnumber, compression="gzip")
    h5.create_dataset("energy", data = energy, compression="gzip")
