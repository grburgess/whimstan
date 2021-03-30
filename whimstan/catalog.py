from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py


@dataclass
class XRTCatalogEntry:

    name: str
    ra: float
    dec: float
    nH_mw: float
    z: float
    nH_host_sim: Optional[float] = None
    index_sim: Optional[float] = None
    flux_sim: Optional[float] = None
    n0_sim: Optional[float] = None
    temp_sim: Optional[float] = None


class XRTCatalog(object):

    def __init__(self, *grbs):

        self._catalog: Dict[str, XRTCatalogEntry] = {}

        for grb in grbs:

            self._catalog[grb.name] = grb

    @property
    def catalog(self):

        return self._catalog

    def to_file(self, file_name):

        with h5py.File(file_name, "w") as f:

            for k, v in self._catalog.items():

                grp = f.create_group(k)
                grp.attrs["ra"] = v.ra
                grp.attrs["dec"] = v.dec
                grp.attrs["z"] = v.z
                grp.attrs["nH_mw"] = v.nH_mw

                if v.nH_host_sim is not None:

                    grp.attrs["nH_host_sim"] = v.nH_host_sim

                if v.index_sim is not None:

                    grp.attrs["index_sim"] = v.index_sim

                if v.flux_sim is not None:

                    grp.attrs["flux_sim"] = v.flux_sim

                if v.n0_sim is not None:

                    grp.attrs["n0_sim"] = v.n0_sim

                if v.temp_sim is not None:

                    grp.attrs["temp_sim"] = v.temp_sim

    @classmethod
    def from_file(cls, file_name):

        with h5py.File(file_name, "r") as f:

            grbs = []

            for k, v in f.items():

                nH_host_sim = None
                index_sim = None
                flux_sim = None
                n0_sim = None
                temp_sim = None
                

                
                if "nH_host_sim" in v.attrs:

                    nH_host_sim = v.attrs["nH_host_sim"]

                if "index_sim" in v.attrs:

                    index_sim = v.attrs["index_sim"]

                if "flux_sim" in v.attrs:

                    flux_sim = v.attrs["flux_sim"]

                if "n0_sim" in v.attrs:

                    n0_sim = v.attrs["n0_sim"]

                if "temp_sim" in v.attrs:

                    temp_sim = v.attrs["temp_sim"]

                tmp = XRTCatalogEntry(
                    name=k,
                    ra=v.attrs["ra"],
                    dec=v.attrs["dec"],
                    z=v.attrs["z"],
                    nH_mw=v.attrs["nH_mw"],
                    nH_host_sim=nH_host_sim,
                    index_sim=index_sim,
                    flux_sim=flux_sim,
                    n0_sim=n0_sim,
                    temp_sim=temp_sim

                )

                grbs.append(tmp)

        return cls(*grbs)
