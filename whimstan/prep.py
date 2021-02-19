from dataclasses import dataclass
from pathlib import Path
from typing import List
import h5py
import astropy.io.fits as fits
import numpy as np
import pandas as pd
from astromodels.utils.data_files import _get_data_file_path
from threeML import OGIPLike, silence_warnings, update_logging_level

silence_warnings()
update_logging_level("WARNING")


@dataclass
class XRTCatalogEntry:

    name: str
    ra: float
    dec: float
    nH_mw: float
    z: float


class XRTCatalog(object):

    def __init__(self, *grbs):

        self._catalog = {}
        
        for grb in grbs:

            self._catalog[grb.name] = grb

    @property
    def catalog(self):

        return self._catalog

    def to_file(self, file_name):


        with h5py.File(file_name, "w") as f:

            for k,v in self._catalog.items():

                grp = f.create_group(k)
                grp.attrs["ra"] = v.ra
                grp.attrs["dec"] = v.dec
                grp.attrs["z"] = v.z
                grp.attrs["nH_mw"] = v.nH_mw
    

    @classmethod
    def from_file(cls, file_name):

        with h5py.File(file_name, "r") as f:

            grbs = []
            
            for k, v in f.items():

                tmp =XRTCatalogEntry(name=k, ra=v.attrs["ra"], dec=v.attrs["dec"], z=v.attrs["z"], nH_mw=v.attrs["nH_mw"])

                grbs.append(tmp)


        return cls(*grbs)

    
def build_tbabs_arg(ene):

    file_name = _get_data_file_path(Path("xsect/xsect_tbabs_wilm.fits"))
    fxs = fits.open(file_name)
    dxs = fxs[1].data
    xsect_ene = dxs["ENERGY"]
    xsect_val = dxs["SIGMA"]

    return np.interp(ene, xsect_ene,  xsect_val)


@dataclass
class XRTObs:

    n_ene: int
    n_chan: int
    ene_avg: np.array
    ene_width: np.array
    obs_count: List[int]
    bkg_count: List[int]
    mask: np.array
    n_chans_used: int
    scale_factor: float
    rsp: np.array
    exposure: float


def extract_xrt_data(plugin):

    plugin.set_active_measurements("0.3-10.")

    n_chans_used = sum(plugin.mask)

    print(n_chans_used)

    mask = np.zeros(len(plugin.mask))

    mask[:n_chans_used] = np.where(plugin.mask)[0] + 1  # plus one for Stan
    print(len(mask))

    n_ene = len(plugin.response.monte_carlo_energies) - 1
    n_chan = len(plugin.response.ebounds) - 1
    rsp = plugin.response.matrix
    scale_factor = plugin.scale_factor

    ene_lo = plugin.response.monte_carlo_energies[:-1]
    ene_hi = plugin.response.monte_carlo_energies[1:]

    ene_avg = (ene_hi + ene_lo) / 2.
    ene_width = ene_hi - ene_lo

    return XRTObs(n_ene=n_ene,
                  n_chan=n_chan,
                  ene_avg=ene_avg,
                  ene_width=ene_width,
                  obs_count=[int(x) for x in plugin.observed_counts],
                  bkg_count=[int(x) for x in plugin.background_counts],
                  rsp=rsp,
                  scale_factor=scale_factor,
                  mask=[int(x) for x in mask],
                  n_chans_used=n_chans_used,
                  exposure=plugin.exposure


                  )


def build_stan_data(*grbs: str, catalog=None):

    
    N_grbs = len(grbs)
    z = []
    nH_mw = []
    exposure_ratio = []
    counts = []
    bkg = []
    mask = []
    n_chans_used = []
    rsp = []

    N_ene = []
    N_chan = []

    pca = []
    pcaz = []
    ene_avg = []
    ene_width = []
    exposure = []
    for grb in grbs:

        row = initial_data.loc[grb]

        z.append(catalog.catalog[grb].z)
        nH_mw.append(catalog.catalog[grb].nH_mw)

        bpath = Path(f"data/grb{grb}")

        o = OGIPLike("xrt",
                     observation=bpath / "apc.pi",
                     background=bpath / "apcback.pi",
                     response=bpath / "apc.rmf",
                     arf_file=bpath / "apc.arf"
                     )

        x = extract_xrt_data(o)

        N_ene.append(x.n_ene)
        N_chan.append(x.n_chan)

        n_chans_used.append(int(x.n_chans_used))

        rsp.append(x.rsp.tolist())
        exposure_ratio.append(x.scale_factor)
        counts.append(x.obs_count)
        bkg.append(x.bkg_count)

        mask.append(x.mask)

        exposure.append(x.exposure)
        ene_avg.append(x.ene_avg.tolist())
        ene_width.append(x.ene_width.tolist())

        p = build_tbabs_arg(x.ene_avg).tolist()
        pz = build_tbabs_arg(x.ene_avg * (1 + catalog.catalog[grb].z))
        pca.append(p)
        pcaz.append(pz.tolist())

    return dict(
        N_grbs=N_grbs,
        N_chan=N_chan[0],
        N_ene=N_ene[0],
        rsp=rsp,
        nH_mw=nH_mw,
        exposure_ratio=exposure_ratio,
        ene_avg=ene_avg,
        ene_width=ene_width,
        counts=counts,
        bkg=bkg,
        mask=mask,
        n_chans_used=n_chans_used,
        z=z,
        precomputed_absorp=pca,
        host_precomputed_absorp=pcaz,

        exposure=exposure



    )
