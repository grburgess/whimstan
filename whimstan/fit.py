from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import arviz as av
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from astromodels import Model, PointSource, Powerlaw, Powerlaw_Eflux, TbAbs
from bb_astromodels import Integrate_Absori
from natsort import natsorted

from .catalog import XRTCatalog


@dataclass
class ModelContainer:

    model_all: Optional[Model] = None
    model_host: Optional[Model] = None
    model_mw: Optional[Model] = None
    model_pl: Optional[Model] = None


class Fit(object):

    def __init__(self, catalog: XRTCatalog, stan_fit: av.data.InferenceData, data_path: Optional[Path] = None):
        """TODO describe function

        :param catalog:
        :type catalog: XRTCatalog
        :param stan_fit:
        :type stan_fit: av.data.InferenceData
        :param data_path:
        :type data_path: Optional[Path]
        :returns:

        """
        self._catalog: XRTCatalog = catalog
        self._data_path: Path = data_path

        self._n_grbs: int = stan_fit.posterior.K.stack(
            sample=("chain", "draw")).values.shape[0]

        self._flux: np.ndarray = stan_fit.posterior.K.stack(
            sample=("chain", "draw")).values
        self._index: np.ndarray = stan_fit.posterior.index.stack(
            sample=("chain", "draw")).values

        self._has_host_fit: bool = False
        self._host_nh: Optional[np.ndarray] = None
        self._log_nh_host_mu: Optional[np.ndarray] = None
        self._log_nh_host_sigma: Optional[np.ndarray] = None

        try:
            self._host_nh = stan_fit.posterior.nH_host_norm.stack(
                sample=("chain", "draw")).values

            self._log_nh_host_mu = stan_fit.posterior.log_nH_host_mu.stack(
                sample=("chain", "draw")).values

            self._log_nh_host_sigma = stan_fit.posterior.log_nH_host_sigma.stack(
                sample=("chain", "draw")).values

            self._has_host_fit = True

        except:

            # we do not have a host gas fit

            pass

        # whim stuff
        self._has_whim_fit: bool = False

        # group properties

        self._index_mu: np.ndarray = stan_fit.posterior.index_mu.stack(
            sample=("chain", "draw")).values

        self._index_sigma: np.ndarray = stan_fit.posterior.index_sigma.stack(
            sample=("chain", "draw")).values

        if data_path is not None:

            self._grbs: Optional[List[str]] = [x.name.replace("grb", "")
                                               for x in natsorted(data_path.glob("grb*"))]

        else:

            self._grbs = None

        # if it is a sim, lets go ahead
        # and see what's in there

        self._has_whim_sim: bool = False
        self._has_host_sim: bool = False

        if self._catalog.is_sim:

            if self._catalog.n0_sim is not None:

                self._has_whim_sim = True

            if self._catalog.nH_host_sim is not None:

                self._has_host_sim = True

    @property
    def catalog(self) -> XRTCatalog:
        self._catalog

    @property
    def n_grbs(self) -> int:
        return self._n_grbs

    @property
    def z(self) -> np.ndarray:
        return self._catalog.z

    @property
    def flux(self) -> np.ndarray:
        return self._flux

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def host_nh(self) -> np.ndarray:
        """
        host nH denisity in 1/10^22 cm2

        :returns:

        """

        return self._host_nh

    @property
    def index_mu(self) -> np.ndarray:
        return self._index_mu

    @property
    def index_sigma(self) -> np.ndarray:
        return self._index_sigma

    @property
    def log_nh_host_mu(self) -> np.ndarray:
        return self._log_nh_host_mu

    @property
    def log_nh_host_sigma(self) -> np.ndarray:
        return self._log_nh_host_sigma

    def plot_nh_host_distribution(self):

        fig, ax = plt.subplots()

        xgrid = np.linspace(19., 25, 100)

        # if we have a simulation
        # then plot the data

        if self._catalog.is_sim:

            ax.hist(np.log10(self._catalog.nH_host_sim) + 22, bins=10,
                    density=True, histtype="step", lw=2)

            # ax.plot(xgrid, stats.norm.pdf(xgrid, loc=, scale=0.5),  color="b")

        for mu, sig in zip(self._index_mu, self._index_sigma):

            ax.plot(xgrid, stats.norm.pdf(
                xgrid, loc=mu, scale=sig), alpha=0.1, color="r")

        ax.set_xlabel("log10(nH host)")

        return fig

    def _get_spectrum(self, id) -> ModelContainer:
        """


        :param id: 
        :type id: 
        :returns: 

        """

        card = self._catalog.catalog[self._grbs[id]]

        model_all: Optional[Model] = None

        if self._has_whim_fit or self._has_whim_sim:

            spec_all = Powerlaw_Eflux(a=.4, b=15) * TbAbs(NH=card.nH_mw) * TbAbs(
                redshift=card.z) * Integrate_Absori(redshift=card.z, n0=card.n0_sim, temp=card.temp_sim)

            spec_all.NH_2.fix = True
            spec_all.xi_4.fix = True
            spec_all.gamma_4.fix = True
            spec_all.abundance_4.fix = True
            spec_all.fe_abundance_4.fix = True

            # if not self._has_whim_fit:

            #     spec_all.n0_4.fix = True

            #     spec_all.temp_4.fix = True

            ps_all = PointSource("all", 0, 0, spectral_shape=spec_all)

            model_all = Model(ps_all)

        model_host: Optional[Model] = None
        model_mw: Optional[Model] = None

        if self._has_host_sim or self._has_host_fit:

            spec_host = Powerlaw_Eflux(
                a=.4, b=15) * TbAbs(NH=card.nH_mw) * TbAbs(redshift=card.z)
            spec_host.NH_2.fix = True

            ps_host = PointSource("host", 0, 0, spectral_shape=spec_host)

            model_host = Model(ps_host)

            spec_mw = Powerlaw_Eflux(a=.14, b=15) * TbAbs(NH=card.nH_mw)
            spec_mw.NH_2.fix = True

            ps_mw = PointSource("mw", 0, 0, spectral_shape=spec_mw)
            model_mw = Model(ps_mw)

        spec_pl = Powerlaw_Eflux(a=.4, b=15)

        ps_pl = PointSource("pl", 0, 0, spectral_shape=spec_pl)
        model_pl = Model(ps_pl)

        return ModelContainer(model_all, model_host, model_mw, model_pl)

    def plot_model_spectrum(self, id):

        # get the model container object
        model_container: ModelContainer = self._get_spectrum(id)

        # if this is a simualtion
        # we want to plot those
        # components

        card = self._catalog.catalog[self._grbs[id]]

        # first, let's plot the simualted parameters

        fig, ax = plt.subplots()

        ene = np.geomspace(0.1, 5, 500)

        if self._has_whim_fit:

            pass

        if self._has_host_fit:

            samples = np.vstack(
                (self._flux[id], self._index[id], self._host_nh[id]))

        for sample in samples.T[::10]:

            if self._has_whim_fit:

                model_container.model_all.set_free_parameters(sample)

                ax.loglog(ene, model_container.model_all.get_point_source_fluxes(
                    0, ene), color="purple", lw=1, alpha=0.1)

            if self._has_host_fit:

                model_container.model_mw.set_free_parameters(sample[:2])

                model_container.model_host.set_free_parameters(sample[:3])

                ax.loglog(ene, model_container.model_mw.get_point_source_fluxes(
                    0, ene), color="red", lw=1, alpha=0.1)

                ax.loglog(ene, model_container.model_host.get_point_source_fluxes(
                    0, ene), color="green", lw=1, alpha=0.1)

            model_container.model_pl.set_free_parameters(sample[:2])

            ax.loglog(ene, model_container.model_pl.get_point_source_fluxes(
                0, ene), color="blue", lw=1, alpha=0.1)

        if self._catalog.is_sim:

            simulated_parameters = card.simulated_parameters

            if model_container.model_all is not None:

                # ok, we have some whim

                model_container.model_all.set_free_parameters(
                    simulated_parameters)

                ax.loglog(ene, model_container.model_all.get_point_source_fluxes(
                    0, ene), color="limegreen")

            if model_container.model_host is not None:

                # ok, we have some host gas (MW as well)

                model_container.model_mw.set_free_parameters(
                    simulated_parameters[:2])

                model_container.model_host.set_free_parameters(
                    simulated_parameters[:3])

                ax.loglog(ene, model_container.model_mw.get_point_source_fluxes(
                    0, ene), color="white")

                ax.loglog(ene, model_container.model_host.get_point_source_fluxes(
                    0, ene), color="grey")

            model_container.model_pl.set_free_parameters(
                simulated_parameters[:2])

            ax.loglog(ene, model_container.model_pl.get_point_source_fluxes(
                0, ene), color="yellow")

        return fig
