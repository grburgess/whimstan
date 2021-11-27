from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


@dataclass
class XRTCatalogEntry:

    name: str
    ra: float
    dec: float
    z: float
    nH_mw: Optional[float] = None
    nH_host_sim: Optional[float] = None
    index_sim: Optional[float] = None
    flux_sim: Optional[float] = None
    n0_sim: Optional[float] = None
    temp_sim: Optional[float] = None

    @property
    def simulated_parameters(self) -> np.ndarray:
        """

        get a numpy array of the simulated parameters

        :returns:

        """
        tmp: List[float] = []

        if self.flux_sim is not None:

            tmp.append(self.flux_sim)

        if self.index_sim is not None:

            tmp.append(self.index_sim)

        if self.nH_host_sim is not None:

            tmp.append(self.nH_host_sim)

        if self.n0_sim is not None:

            tmp.append(self.n0_sim)

        if self.temp_sim is not None:

            tmp.append(self.temp_sim)

        return np.array(tmp)


class XRTCatalog:
    def __init__(self, *grbs):

        self._catalog: Dict[str, XRTCatalogEntry] = OrderedDict()

        for grb in grbs:

            self._catalog[grb.name] = grb

        # create a flag if this is a sim

        if grb.nH_host_sim is None:

            self._is_sim = False

        else:

            self._is_sim = True

    def get_sub_selection(self, selection: np.ndarray) -> "XRTCatalog":
        """
        pass in a selection boolean array and this builds a sub catalog
        """

        out: List[XRTCatalogEntry] = []

        for (k, v), flag in zip(self._catalog.items(), selection):

            if flag:

                out.append(v)

        return XRTCatalog(*out)

    @property
    def grbs(self) -> List[str]:

        return list(self._catalog.keys())

    @property
    def is_sim(self) -> bool:
        """
        if this is from a simulation
        """

        return self._is_sim

    @property
    def z(self) -> np.ndarray:
        z = []

        for k, v in self._catalog.items():

            z.append(v.z)

        return np.array(z)

    @property
    def ra(self) -> np.ndarray:
        ra = []

        for k, v in self._catalog.items():

            ra.append(v.ra)

        return np.array(ra)

    @property
    def dec(self) -> np.ndarray:
        dec = []

        for k, v in self._catalog.items():

            dec.append(v.dec)

        return np.array(dec)

    @property
    def nH_mw(self) -> np.ndarray:
        nH_mw = []

        for k, v in self._catalog.items():

            nH_mw.append(v.nH_mw)

        return np.array(nH_mw)

    @property
    def nH_host_sim(self) -> Optional[np.ndarray]:
        if self._is_sim:
            nH_host_sim = []

            for k, v in self._catalog.items():

                nH_host_sim.append(v.nH_host_sim)

            return np.array(nH_host_sim)

        else:

            return None

    @property
    def index_sim(self) -> Optional[np.ndarray]:
        if self._is_sim:
            index_sim = []

            for k, v in self._catalog.items():

                index_sim.append(v.index_sim)

            return np.array(index_sim)

        else:

            return None

    @property
    def flux_sim(self) -> Optional[np.ndarray]:
        if self._is_sim:
            flux_sim = []

            for k, v in self._catalog.items():

                flux_sim.append(v.flux_sim)

            return np.array(flux_sim)

        else:

            return None

    @property
    def n0_sim(self) -> Optional[float]:

        if self._is_sim:

            # just grab the first element as
            # they are all the same

            return list(self._catalog.values())[0].n0_sim

        else:

            return None

    @property
    def temp_sim(self) -> Optional[float]:

        if self._is_sim:

            # just grab the first element as
            # they are all the same

            return list(self._catalog.values())[0].temp_sim

        else:

            return None

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

                if v.nH_mw is not None:

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
                nH_mw = None

                if "nH_mw" in v.attrs:

                    nH_mw = v.attrs["nH_mw"]

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
                    nH_mw=nH_mw,
                    nH_host_sim=nH_host_sim,
                    index_sim=index_sim,
                    flux_sim=flux_sim,
                    n0_sim=n0_sim,
                    temp_sim=temp_sim,
                )

                grbs.append(tmp)

        return cls(*grbs)
