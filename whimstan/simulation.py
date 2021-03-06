
from pathlib import Path
from typing import Dict, List, Optional

import astropy.units as u
import popsynth
from astromodels import Model, PointSource, Powerlaw_Eflux, TbAbs
from astropy.coordinates import SkyCoord
from bb_astromodels import Integrate_Absori
from gdpyc import GasMap
from popsynth.utils.progress_bar import progress_bar
from threeML import OGIPLike, quiet_mode

from .catalog import XRTCatalog, XRTCatalogEntry


class SpectrumGenerator(object):

    def __init__(self, name: str, eflux: float, index: float, ra: float, dec: float, z: float, host_nh: float, mw_nh: float, whim_n0: Optional[float] = None, whim_T: Optional[float] = None, demo_plugin: Optional[OGIPLike] = None,
                 use_mw_gas: bool = True, use_host_gas: bool = True):
        """

        :param name:  the name of the function
        :type name: str
        :param eflux: 
        :type eflux: float
        :param index: 
        :type index: float
        :param ra: 
        :type ra: float
        :param dec: 
        :type dec: float
        :param z: 
        :type z: float
        :param host_nh: host nH density in 1/10^22 cm2
        :type host_nh: float
        :param mw_nh: milkyway nH density in 1/10^22 cm2
        :type mw_nh: float
        :param whim_n0: 
        :type whim_n0: Optional[float]
        :param whim_T: 
        :type whim_T: Optional[float]
        :param demo_plugin: 
        :type demo_plugin: Optional[OGIPLike]
        :param use_mw_gas: 
        :type use_mw_gas: bool
        :param use_host_gas: 
        :type use_host_gas: bool
        :returns: 

        """

        self._name: str = name
        self._eflux: float = eflux
        self._index: float = index
        self._ra: float = ra
        self._dec: float = dec
        self._z: float = z
        self._host_nh: float = host_nh
        self._mw_nh: float = mw_nh
        self._whim_n0: Optional[float] = whim_n0
        self._whim_T: Optional[float] = whim_T
        self._use_mw_gas: bool = use_mw_gas
        self._use_host_gas: bool = use_host_gas

        self._demo_plugin: Optional[OGIPLike] = demo_plugin

        # now this is done in pop synth
#        self._get_mw_nh()

        self._create_plugin()

    def _get_mw_nh(self):

        coord = SkyCoord(ra=self._ra, dec=self._dec, unit="deg", frame="icrs")

        self._mw_nh = GasMap.nhf(
            coord, nhmap='DL', radius=1*u.deg).value / 1e22

    def _create_plugin(self):
        quiet_mode()

        if self._demo_plugin is None:

            self._demo_plugin = OGIPLike("tmp",
                                         observation="data/grb050401/apcsource.pi",
                                         background="data/grb050401/apcback.pi",
                                         response="data/grb050401/apc.rmf",
                                         arf_file="data/grb050401/apc.arf",
                                         verbose=False

                                         )

        spec = Powerlaw_Eflux(F=self._eflux, index=self._index, a=.4, b=15)
        if self._use_mw_gas:
            spec *= TbAbs(NH=self._mw_nh, redshift=0)
        if self._use_host_gas:
            spec *= TbAbs(NH=self._host_nh, redshift=self._z)

        if (self._whim_n0 is not None) and (self._whim_T is not None):

            spec = spec * \
                Integrate_Absori(n0=self._whim_n0,
                                 temp=self._whim_T, redshift=self._z)

        ps = PointSource("tmp", self._ra, self._dec, spectral_shape=spec)
        model = Model(ps)

        self._demo_plugin.set_model(model)

        simulation = self._demo_plugin.get_simulated_dataset()

        self._simulated_data = simulation

    @property
    def name(self):
        return self._name

    @property
    def simulated_data(self):
        return self._simulated_data

    @property
    def xrt_catalog_entry(self):

        return XRTCatalogEntry(self._name.replace("grb", ""),
                               self._ra,
                               self._dec,
                               self._z,
                               self._mw_nh,
                               nH_host_sim=self._host_nh,
                               index_sim=self._index,
                               flux_sim=self._eflux,
                               n0_sim=self._whim_n0,
                               temp_sim=self._whim_T


                               )


class SpectrumFactory(object):

    def __init__(self, population: popsynth.Population, whim_n0=None, whim_T=None,
                 use_mw_gas=True, use_host_gas=True):

        self._spectra = []

        for i in progress_bar(range(population.n_objects), desc="Calculating the simulated datasets"):

            name = f"grb00{i}"

            if use_mw_gas:
                mw_nh = population.mw_nh[i]/1.e22
            else:
                mw_nh = None

            if use_host_gas:
                host_nh = population.host_nh[i]/1.e22
            else:
                host_nh = None

            sg = SpectrumGenerator(name=name,
                                   eflux=population.fluxes_latent[i],
                                   index=population.spec_idx[i],
                                   ra=population.ra[i],
                                   dec=population.dec[i],
                                   z=population.distances[i],
                                   host_nh=host_nh,
                                   mw_nh=mw_nh,
                                   whim_n0=whim_n0,
                                   whim_T=whim_T,
                                   use_mw_gas=use_mw_gas,
                                   use_host_gas=use_host_gas)

            self._spectra.append(sg)

    def write_data(self, path="data", catalog_name="sim_cat.h5"):

        cat_entries = []

        root = Path(path)

        root.mkdir(parents=True, exist_ok=True)

        for s in self._spectra:

            p = root / s.name
            p.mkdir(parents=True, exist_ok=True)

            pi: OGIPLike = s.simulated_data
            pi.write_pha(p / "apc", force_rsp_write=True)

            cat_entries.append(s.xrt_catalog_entry)

        xrt_cat = XRTCatalog(*cat_entries)

        xrt_cat.to_file(catalog_name)

    @property
    def spectra(self) -> List:
        return self._spectra
