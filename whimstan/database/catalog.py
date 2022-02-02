from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

import healpy as hp
from astropy.coordinates import SkyCoord

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from astromodels import Model, PointSource, Powerlaw_Eflux, TbAbs
from bb_astromodels import Integrate_Absori

import h5py
import numpy as np

from ..utils import (
    get_path_of_data_file,
    Colors,
    build_custom_continuous_cmap,
    hex_to_rgb,
)

from ..utils.projections import *

@dataclass
class ModelContainer:

    model_all: Optional[Model] = None
    model_host: Optional[Model] = None
    model_mw: Optional[Model] = None
    model_pl: Optional[Model] = None


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
    arf: np.array
    exposure: float

    @classmethod
    def extract_xrt_data(cls, plugin: DispersionSpectrumLike):

        plugin.set_active_measurements("0.3-10.")

        n_chans_used = sum(plugin.mask)

        mask = np.zeros(len(plugin.mask))

        mask[:n_chans_used] = np.where(plugin.mask)[0] + 1  # plus one for Stan

        n_ene = len(plugin.response.monte_carlo_energies) - 1
        n_chan = len(plugin.response.ebounds) - 1
        rsp = plugin.response.rmf
        arf = plugin.response.arf
        scale_factor = plugin.scale_factor

        ene_lo = plugin.response.monte_carlo_energies[:-1]
        ene_hi = plugin.response.monte_carlo_energies[1:]

        ene_avg = (ene_hi + ene_lo) / 2.0
        ene_width = ene_hi - ene_lo

        return cls(
            n_ene=n_ene,
            n_chan=n_chan,
            ene_avg=ene_avg,
            ene_width=ene_width,
            obs_count=[int(x) for x in plugin.observed_counts],
            bkg_count=[int(x) for x in plugin.background_counts],
            rsp=rsp,
            arf=arf,
            scale_factor=float(scale_factor),
            mask=[int(x) for x in mask],
            n_chans_used=int(n_chans_used),
            exposure=float(plugin.exposure),
        )


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

    def get_spectrum(
        self, with_whim: bool = True, with_host: bool = True
    ) -> ModelContainer:
        """


        :param id:
        :type id:
        :returns:

        """

        model_all: Optional[Model] = None

        if with_whim:

            # if there is no simulation,
            # then we will setup with dummy parameters
            n0 = 1e-7
            temp = 1e6

            if self.temp_sim is not None:

                n0 = self.n0_sim
                temp = self.temp_sim

            spec_all = (
                Powerlaw_Eflux(a=0.4, b=15)
                * TbAbs(NH=self.nH_mw)
                * TbAbs(redshift=self.z)
                * Integrate_Absori(redshift=self.z, n0=n0, temp=temp)
            )

            # fix the things we do not vary

            spec_all.NH_2.fix = True
            spec_all.xi_4.fix = True
            spec_all.gamma_4.fix = True
            spec_all.abundance_4.fix = True
            spec_all.fe_abundance_4.fix = True

            # kill all the bounds

            for k, v in spec_all.parameters.items():

                v.bounds = (None, None)

            ps_all = PointSource("all", 0, 0, spectral_shape=spec_all)

            model_all = Model(ps_all)

        model_host: Optional[Model] = None
        model_mw: Optional[Model] = None

        if with_host:

            spec_host = (
                Powerlaw_Eflux(a=0.4, b=15)
                * TbAbs(NH=self.nH_mw)
                * TbAbs(redshift=self.z)
            )
            spec_host.NH_2.fix = True

            for k, v in spec_host.parameters.items():

                v.bounds = (None, None)

            ps_host = PointSource("host", 0, 0, spectral_shape=spec_host)

            model_host = Model(ps_host)

            spec_mw = Powerlaw_Eflux(a=0.14, b=15) * TbAbs(NH=self.nH_mw)
            spec_mw.NH_2.fix = True

            for k, v in spec_mw.parameters.items():

                v.bounds = (None, None)

            ps_mw = PointSource("mw", 0, 0, spectral_shape=spec_mw)
            model_mw = Model(ps_mw)

        spec_pl = Powerlaw_Eflux(a=0.4, b=15)

        for k, v in spec_pl.parameters.items():

            v.bounds = (None, None)

        ps_pl = PointSource("pl", 0, 0, spectral_shape=spec_pl)
        model_pl = Model(ps_pl)

        return ModelContainer(model_all, model_host, model_mw, model_pl)


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

    def write(self, file_name):

        """
        write the catalog to a file or hdf group

        :param file_name:
        :type file_name:
        :returns:

        """
        if isinstance(file_name, str):

            is_file = True

            f = h5py.File(file_name, "w")

        elif isinstance(file_name, h5py.Group):

            f = file_name

            is_file = False

        else:

            raise RuntimeError()

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

        if is_file:

            f.close()

    @classmethod
    def from_file(cls, file_name):

        if isinstance(file_name, str):

            is_file = True

            f = h5py.File(file_name, "r")

        elif isinstance(file_name, h5py.Group):

            f = file_name

            is_file = False

        else:

            raise RuntimeError()

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

        if is_file:

            f.close()

        return cls(*grbs)

    def plot_skymap(self, mw_limit: float = 1e21) -> plt.Figure:

        # open the file

        with h5py.File(get_path_of_data_file("mw_map.h5"), "r") as f:

            gas_map: np.ndarray = f["map"][()]

        mask = gas_map * 1e22 > mw_limit

        gas_mask = hp.ma(gas_map)

        gas_mask.mask = mask

        fig, ax = plt.subplots(
            subplot_kw=dict(projection="galactic degrees mollweide")
        )

        ax.grid()

        new_cmap = build_custom_continuous_cmap(
            hex_to_rgb(Colors.black), hex_to_rgb(Colors.purple)
        )

        new_cmap.set_under(Colors.black)

        ax.imshow_hpx(gas_mask.filled(-999), cmap=new_cmap, vmin=0)

        cc = SkyCoord(ra=self.ra, dec=self.dec, unit="deg", frame="icrs")

        ax.scatter(
            cc.galactic.l.deg,
            cc.galactic.b.deg,
            transform=ax.get_transform("galactic"),
            c=Colors.green,
            s=5,
        )

        return fig
