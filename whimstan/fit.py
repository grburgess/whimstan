from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as av
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from astromodels import Model, PointSource, Powerlaw_Eflux, TbAbs
from bb_astromodels import Integrate_Absori
from matplotlib.lines import Line2D
from natsort import natsorted
from numpy.typing import ArrayLike
from threeML import OGIPLike

from .catalog import XRTCatalog, XRTCatalogEntry
from .spectral_plot import display_posterior_model_counts

green = "#00D584"
purple = "#985CFC"
yellow = "#EDE966"
grey = "#385656"
lightgrey = "#839393"
black = "#1F2222"


@dataclass
class ModelContainer:

    model_all: Optional[Model] = None
    model_host: Optional[Model] = None
    model_mw: Optional[Model] = None
    model_pl: Optional[Model] = None


class Fit:
    def __init__(
        self,
        catalog: XRTCatalog,
        stan_fit: av.data.InferenceData,
        data_path: Optional[Path] = None,
    ):
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
        self._data_path: Optional[Path] = data_path

        self._n_grbs: int = stan_fit.posterior.K.stack(
            sample=("chain", "draw")
        ).values.shape[0]

        self._flux: ArrayLike = stan_fit.posterior.K.stack(
            sample=("chain", "draw")
        ).values
        self._index: ArrayLike = stan_fit.posterior.index.stack(
            sample=("chain", "draw")
        ).values

        self._has_host_fit: bool = False
        self._host_nh: Optional[ArrayLike] = None
        self._log_nh_host_mu: Optional[ArrayLike] = None
        self._log_nh_host_sigma: Optional[ArrayLike] = None

        self._nh_host_alpha = None

        try:

            self._nh_host_alpha = stan_fit.posterior.host_alpha.stack(
                sample=("chain", "draw")
            ).values

        except:

            pass

        try:
            self._host_nh = stan_fit.posterior.nH_host_norm.stack(
                sample=("chain", "draw")
            ).values

            self._log_nh_host_mu = stan_fit.posterior.log_nH_host_mu_raw.stack(
                sample=("chain", "draw")
            ).values

            self._log_nh_host_sigma = (
                stan_fit.posterior.log_nH_host_sigma.stack(
                    sample=("chain", "draw")
                ).values
            )

            self._has_host_fit = True

        except:

            # we do not have a host gas fit

            pass

        self._has_whim_fit: bool = False
        try:
            self._n0_whim = stan_fit.posterior.n0_whim.stack(
                sample=("chain", "draw")
            ).values
            self._t_whim = stan_fit.posterior.t_whim.stack(
                sample=("chain", "draw")
            ).values
            self._has_whim_fit = True
        except:

            # we do not have a whim fit

            pass

        # group properties

        self._index_mu: ArrayLike = stan_fit.posterior.index_mu.stack(
            sample=("chain", "draw")
        ).values

        self._index_sigma: ArrayLike = stan_fit.posterior.index_sigma.stack(
            sample=("chain", "draw")
        ).values

        if data_path is not None:

            self._grbs: Optional[List[str]] = [
                x.name.replace("grb", "")
                for x in natsorted(data_path.glob("grb*"))
            ]

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
        """
        The catalog of the GRBs

        :returns:

        """
        self._catalog

    @property
    def n_grbs(self) -> int:
        """
        The number of GRBs in the fit

        :returns:

        """

        return self._n_grbs

    @property
    def z(self) -> ArrayLike:
        return self._catalog.z

    @property
    def flux(self) -> ArrayLike:
        return self._flux

    @property
    def index(self) -> ArrayLike:
        return self._index

    @property
    def host_nh(self) -> ArrayLike:
        """
        host nH denisity in 1/10^22 cm2

        :returns:

        """

        return self._host_nh

    @property
    def index_mu(self) -> ArrayLike:
        return self._index_mu

    @property
    def index_sigma(self) -> ArrayLike:
        return self._index_sigma

    @property
    def log_nh_host_mu(self) -> np.ndarray:
        return self._log_nh_host_mu

    @property
    def log_nh_host_sigma(self) -> ArrayLike:
        return self._log_nh_host_sigma

    def plot_nh_host_distribution(self) -> plt.Figure:

        fig, ax = plt.subplots()

        xgrid = np.linspace(19.0, 25, 100)

        # if we have a simulation
        # then plot the data

        if self._catalog.is_sim:

            ax.hist(
                np.log10(self._catalog.nH_host_sim) + 22,
                bins=10,
                density=True,
                histtype="step",
                lw=2,
                color=grey,
            )

            # ax.plot(xgrid, stats.norm.pdf(xgrid, loc=, scale=0.5),  color="b")

        if self._nh_host_alpha is None:

            for mu, sig in zip(
                self._log_nh_host_mu + 22.0, self._log_nh_host_sigma
            ):

                ax.plot(
                    xgrid,
                    stats.norm.pdf(xgrid, loc=mu, scale=sig),
                    alpha=0.1,
                    color=green,
                )

        else:

            for mu, sig, alpha in zip(
                self._log_nh_host_mu + 22.0,
                self._log_nh_host_sigma,
                self._nh_host_alpha,
            ):

                ax.plot(
                    xgrid,
                    stats.skewnorm.pdf(xgrid, a=alpha, loc=mu, scale=sig),
                    alpha=0.1,
                    color=green,
                )

        ax.set_xlabel("log10(nH host)")

        return fig

    def _get_spectrum(self, id: int) -> ModelContainer:
        """


        :param id:
        :type id:
        :returns:

        """

        card: XRTCatalogEntry = self._catalog.catalog[self._grbs[id]]

        model_all: Optional[Model] = None

        if self._has_whim_fit or self._has_whim_sim:

            spec_all = (
                Powerlaw_Eflux(a=0.4, b=15)
                * TbAbs(NH=card.nH_mw)
                * TbAbs(redshift=card.z)
                * Integrate_Absori(
                    redshift=card.z, n0=card.n0_sim, temp=card.temp_sim
                )
            )

            spec_all.NH_2.fix = True
            spec_all.xi_4.fix = True
            spec_all.gamma_4.fix = True
            spec_all.abundance_4.fix = True
            spec_all.fe_abundance_4.fix = True

            # if not self._has_whim_fit:

            #     spec_all.n0_4.fix = True

            #     spec_all.temp_4.fix = True

            for k, v in spec_all.parameters.items():

                v.bounds = (None, None)

            ps_all = PointSource("all", 0, 0, spectral_shape=spec_all)

            model_all = Model(ps_all)

        model_host: Optional[Model] = None
        model_mw: Optional[Model] = None

        if self._has_host_sim or self._has_host_fit:

            spec_host = (
                Powerlaw_Eflux(a=0.4, b=15)
                * TbAbs(NH=card.nH_mw)
                * TbAbs(redshift=card.z)
            )
            spec_host.NH_2.fix = True

            for k, v in spec_host.parameters.items():

                v.bounds = (None, None)

            ps_host = PointSource("host", 0, 0, spectral_shape=spec_host)

            model_host = Model(ps_host)

            spec_mw = Powerlaw_Eflux(a=0.14, b=15) * TbAbs(NH=card.nH_mw)
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

    def _nH_sim_difference(self, id) -> ArrayLike:

        if not self._catalog.is_sim:

            return

        card: XRTCatalogEntry = self._catalog.catalog[self._grbs[id]]

        difference = self._host_nh[id] - card.nH_host_sim

        return difference

    def plot_nH_z_excess(self) -> None:

        mean_difference = np.empty(self._n_grbs)

        for i in range(self._n_grbs):

            mean_difference[i] = np.median(self._nH_sim_difference(i))

        fig, ax = plt.subplots()

        ax.semilogy(self._catalog.z + 1, mean_difference, ".")

        ax.set_ylabel(r"nH $10^{22}$")

        ax.set_xlabel("z+1")

        return fig

    def plot_nH_z(self, show_truth: bool = False) -> plt.Figure:

        fig, ax = plt.subplots()

        if show_truth:

            ax.loglog(
                self._catalog.z + 1,
                1e22 * self._catalog.nH_host_sim,
                "o",
                color=green,
                alpha=1.0,
                zorder=-1000,
            )

        for i in range(self._n_grbs):

            lo, hi = av.hdi(1e22 * self._host_nh[i], hdi_prob=0.95)

            ax.vlines(
                self._catalog.z[i] + 1, lo, hi, color=purple, linewidth=0.7
            )

        ax.set_ylabel(r"host nH (cm$^{-2}$)")

        ax.set_xlabel("z+1")

        ax.set_xscale("log")

        ax.set_yscale("log")

        ax.set_xlim(right=10)

        return fig

    def get_plugin_and_model(self, id: int) -> Tuple[OGIPLike, Model]:

        if self._data_path is not None:

            bpath = self._data_path / f"grb{self._grbs[id]}/"

        else:

            return

        plugin = OGIPLike(
            "xrt",
            observation=bpath / "apc.pha",
            background=bpath / "apc_bak.pha",
            response=bpath / "apc.rsp",
            spectrum_number=1,
        )

        # get the model container object
        model_container: ModelContainer = self._get_spectrum(id)

        if self._has_whim_fit:

            model = model_container.model_all

        elif self._has_host_fit:

            model = model_container.model_host

        return plugin, model

    def plot_data_spectrum(
        self,
        id: int,
        min_rate: float = -99,
        model_color=green,
        data_color=purple,
        thin=2,
    ) -> plt.Figure:

        o, model = self.get_plugin_and_model(id)

        if o is None:

            return

        if self._has_whim_fit:

            samples = np.vstack(
                (
                    self._flux[id],
                    self._index[id],
                    self._host_nh[id],
                    self._n0_whim,
                    self._t_whim,
                )
            )

        elif self._has_host_fit:

            samples = np.vstack(
                (self._flux[id], self._index[id], self._host_nh[id])
            )

        fig = display_posterior_model_counts(
            o,
            model,
            samples.T[::thin],
            shade=False,
            min_rate=min_rate,
            model_color=model_color,
            data_color=data_color,
            # background_color=blue,
            show_background=False,
            source_only=False,
        )

        ax = fig.get_axes()[0]

        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(0.3)

        return fig

    def plot_model_spectrum(self, id: int, thin=2) -> plt.Figure:

        # get the model container object
        model_container: ModelContainer = self._get_spectrum(id)

        # if this is a simualtion
        # we want to plot those
        # components

        card = self._catalog.catalog[self._grbs[id]]

        # first, let's plot the simualted parameters

        fig, ax = plt.subplots()

        ene = np.geomspace(0.1, 5, 500)

        custom_lines = []
        labels = []
        if self._has_whim_fit:
            labels.append("Host and whim posterior")
            custom_lines.append(Line2D([0], [0], color=purple, lw=2))
        if self._has_host_fit:
            labels.append("Host posterior")
            custom_lines.append(Line2D([0], [0], color=green, lw=2))

        if self._has_whim_fit:

            samples = np.vstack(
                (
                    self._flux[id],
                    self._index[id],
                    self._host_nh[id],
                    self._n0_whim,
                    self._t_whim,
                )
            )

        elif self._has_host_fit:

            samples = np.vstack(
                (self._flux[id], self._index[id], self._host_nh[id])
            )

        for sample in samples.T[::thin]:

            if self._has_whim_fit:

                model_container.model_all.set_free_parameters(sample)

                ax.loglog(
                    ene,
                    model_container.model_all.get_point_source_fluxes(0, ene),
                    color=purple,
                    lw=1,
                    alpha=0.1,
                )

            if self._has_host_fit:
                model_container.model_mw.set_free_parameters(sample[:2])

                model_container.model_host.set_free_parameters(sample[:3])

                # ax.loglog(ene, model_container.model_mw.get_point_source_fluxes(
                #     0, ene), color="red", lw=1, alpha=0.1)

                ax.loglog(
                    ene,
                    model_container.model_host.get_point_source_fluxes(0, ene),
                    color=green,
                    lw=1,
                    alpha=0.1,
                )

            model_container.model_pl.set_free_parameters(sample[:2])

            ax.loglog(
                ene,
                model_container.model_pl.get_point_source_fluxes(0, ene),
                color=yellow,
                lw=1,
                alpha=0.1,
            )

        if self._catalog.is_sim:

            simulated_parameters = card.simulated_parameters

            if model_container.model_all is not None:

                # ok, we have some whim
                labels.append("Total Simualted")
                custom_lines.append(Line2D([0], [0], color=black, lw=2))
                model_container.model_all.set_free_parameters(
                    simulated_parameters
                )

                ax.loglog(
                    ene,
                    model_container.model_all.get_point_source_fluxes(0, ene),
                    color=lightgrey,
                    lw=0.5,
                )

            if model_container.model_host is not None:

                # ok, we have some host gas (MW as well)
                labels.append("Simulation Host included")
                custom_lines.append(Line2D([0], [0], color=grey, lw=2))
                # labels.append("Simulation MW included")
                # custom_lines.append(Line2D([0], [0], color=red, lw=2))
                model_container.model_mw.set_free_parameters(
                    simulated_parameters[:2]
                )

                model_container.model_host.set_free_parameters(
                    simulated_parameters[:3]
                )

                # ax.loglog(ene, model_container.model_mw.get_point_source_fluxes(
                #     0, ene), color="white")

                ax.loglog(
                    ene,
                    model_container.model_host.get_point_source_fluxes(0, ene),
                    color=grey,
                    lw=0.5,
                )

            model_container.model_pl.set_free_parameters(
                simulated_parameters[:2]
            )

            ax.loglog(
                ene,
                model_container.model_pl.get_point_source_fluxes(0, ene),
                color=black,
                lw=0.5,
            )
        ax.legend(custom_lines, labels)

        ax.set_xlabel("energy (keV)")
        ax.set_ylabel(r"flux phts s$^{-1}$kev$^{-1}$cm$^{-2}$)")

        return fig