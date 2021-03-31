import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

absori_elements = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Fe"]
def get_abundance(name="angr"):
    with open("/data/bbiltzing/sw/whimstan/whimstan/data/abundances.dat") as f:
        rows = f.readlines()
        ele = np.array(rows[0].split(" "), dtype=str)
        ele = ele[ele!=""][1:]
        # get rid of \n at the end
        ele[-1] = ele[-1][:2]
        vals = np.zeros((7, len(ele)))
        keys = []
        for i, row in enumerate(rows[1:8]):
            l = np.array(row.split(" "), dtype=str)
            l = l[l!=""]
            # get rid of \n at the end
            if l[-1][-2:]=="\n":
                l[-1] = l[-1][:2]
            if l[-1]=="\n":
                l = l[:-1]
            vals[i] = np.array(l[1:], dtype=float)
            keys.append(l[0][:-1])
        keys = np.array(keys)
    vals_all = np.zeros(len(absori_elements))
    for i, element in enumerate(absori_elements):
        assert element in ele, f"{element} not a valid element. Valid elements: {ele}"

        idx = np.argwhere(ele==element)[0,0]

        assert name in keys, f"{name} not a valid name. Valid names: {keys}"

        idy = np.argwhere(keys==name)[0,0]

        vals_all[i]=vals[idy, idx]

    return vals_all

def load_absori_base():
    ion = np.zeros((10, 26, 10))
    sigma = np.zeros((10, 26, 721))
    atomicnumber = np.empty(10, dtype=int)

    with fits.open("/data/bbiltzing/sw/whimstan/whimstan/data/mansig.fits") as f:
        znumber = f["SIGMAS"].data["Z"]
        ionnumber =  f["SIGMAS"].data["ION"]
        sigmadata =  f["SIGMAS"].data["SIGMA"]
        iondata =  f["SIGMAS"].data["IONDATA"]

        energy =  f["ENERGIES"].data["ENERGY"]

    currentZ = -1
    iZ=-1
    iIon=-1
    for i in range(len(znumber)):
        if znumber[i]!=currentZ:
            iZ+=1
            atomicnumber[iZ] = znumber[i]
            currentZ = znumber[i]
            iIon = -1
        iIon+=1
        for k in range(10):
            ion[iZ,iIon,k] = iondata[i][k]

        # change units of coef

        ion[iZ][iIon][1] *= 1.0E+10;
        ion[iZ][iIon][3] *= 1.0E+04;
        ion[iZ][iIon][4] *= 1.0E-04;
        ion[iZ][iIon][6] *= 1.0E-04;

        for k in range(721):
            sigma[iZ][iIon][k] = sigmadata[i][k]/6.6e-27

    elementname = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Fe"]

    ion = ion
    sigma = sigma
    atomicnumber = atomicnumber
    energy = energy

    return ion, sigma, atomicnumber, energy


def get_spec(gamma=2):
    assert gamma==2, "Only for gamma=2 at the moment"
    return np.load("/data/bbiltzing/sw/whimstan/whimstan/data/spec_gamma2.npy")


def interpolate_sigma(ekeV, energy_base, sigma_base):
    e=1000*ekeV
    res = np.zeros((e.shape[0], e.shape[1], 26, 10))
    mask1 = e>energy_base[-1]
    mask2 = e<energy_base[0]
    mask3 = (~mask1)*(~mask2)
    
    sigma_interp = interp1d(energy_base, sigma_base, axis=0)
    res[mask3]=sigma_interp(e[mask3])
    
    res[mask1] = np.expand_dims(sigma_base[720], axis=0)
    res[mask1] *= np.expand_dims(np.power((e[mask1]/energy_base[-1]),-3.0), axis=(1,2))

    res[mask2] = np.expand_dims(sigma_base[0], axis=0)
    return res


# Constants
omegam=0.3
omegal=0.7
h0=70.
c=2.99792458e5
cmpermpc=3.08568e24
def sum_sigma_interp_precalc(z, x, energy_base, sigma_base, zshell_thickness=0.02):
    nz = int(z/zshell_thickness)
    zsam = z/nz
    zz=zsam*0.5

    # all the different redshifted energies in the
    # z shells
    energy_z = np.zeros((len(x), nz))
    # weight factors from z integral and constants
    zf = np.zeros(nz)

    # loop through shells
    for i in range(nz):
        z1 = zz+1
        energy_z[:,i]=z1*x
        zf[i]=(z1**2/np.sqrt(omegam*(z1**3)+omegal))
        zz+=zsam
    zf*=zsam*c*cmpermpc/h0*6.6e-5*1e-22
    sigma_inter = interpolate_sigma(energy_z, energy_base, sigma_base)
    sigma_inter = np.swapaxes(sigma_inter, 0, 1)
    sigma_inter = np.swapaxes(sigma_inter, 2,3)

    return np.sum(sigma_inter.T*zf, axis=-1).T
    
