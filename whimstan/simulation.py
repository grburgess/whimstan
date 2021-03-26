
from pathlib import Path
from typing import List

import astropy.units as u
import numba as nb
import numpy as np
import popsynth
from astromodels import Model, PointSource, Powerlaw_Eflux, TbAbs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
from bb_astromodels import Integrate_Absori
from gdpyc import DustMap, GasMap
from threeML import (DataList, JointLikelihood, OGIPLike,
                     display_spectrum_model_counts, quiet_mode)

from .prep import XRTCatalog, XRTCatalogEntry


class SpectrumGenerator(object):

    def __init__(self, name, eflux, index, ra, dec, z, host_nh, whim_n0=None, whim_T=None, demo_plugin=None):

        self._name = name
        self._eflux = eflux
        self._index = index
        self._ra = ra
        self._dec = dec
        self._z = z
        self._host_nh = host_nh
        self._mw_nh = None
        self._whim_n0 = whim_n0
        self._whim_T = whim_T

        self._demo_plugin = demo_plugin

        self._get_mw_nh()

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

        spec = Powerlaw_Eflux(F=self._eflux, index=self._index, a=.4, b=15) * TbAbs(NH=self._mw_nh,
                                                                                    redshift=0) * TbAbs(NH=self._host_nh, redshift=self._z)

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
    def xrt_catalof_entry(self):

        return XRTCatalogEntry(self._name.replace("grb", ""), self._ra, self._dec, self._mw_nh, self._z)


class SpectrumFactory(object):

    def __init__(self, population: popsynth.Population, whim_n0=None, whim_T=None):

        self._spectra = []

        for i in range(population.n_objects):

            name = f"grb00{i}"

            sg = SpectrumGenerator(name=name,
                                   eflux=population.fluxes_latent[i],
                                   index=population.spec_idx[i],
                                   ra=population.ra[i],
                                   dec=population.dec[i],
                                   z=population.distances[i],
                                   host_nh=population.host_nh[i]/1.e22,
                                   whim_n0=whim_n0,
                                   whim_T=whim_T)

            self._spectra.append(sg)

    def write_data(self, path="data"):

        cat_entries = []

        root = Path(path)

        root.mkdir(parents=True, exist_ok=True)

        for s in self._spectra:

            p = root / s.name
            p.mkdir(parents=True, exist_ok=True)

            pi: OGIPLike = s.simulated_data
            pi.write_pha(p / "apc", force_rsp_write=True)

            cat_entries.append(s.xrt_catalof_entry)

        xrt_cat = XRTCatalog(*cat_entries)

        xrt_cat.to_file("sim_cat.h5")

    @property
    def spectra(self) -> List:
        return self._spectra
