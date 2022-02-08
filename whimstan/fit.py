from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import arviz as av
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from astromodels import Model
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from threeML.plugins.OGIPLike import OGIPLike
from twopc import compute_postpc

from .database import Database, ModelContainer, XRTCatalog, XRTCatalogEntry
from .spectral_plot import display_posterior_model_counts
from .utils import Colors, build_custom_continuous_cmap
from .utils.array_to_cmap import array_to_cmap
from .utils.dist_plotter import dist_plotter
from .utils.stats_tools import Quantile, extract_quantiles, rank_quantile_width


@dataclass
class HyperObject:
    """
    Contains the parameters from distributions that
    come from hyper parameters for ease of plotting
    """

    mu: np.ndarray
    sigma: np.ndarray
    object_level: np.ndarray
    n_samples: int
    alpha: Optional[np.ndarray] = None
    simulated_parameter: Optional[np.ndarray] = None

    def _distribution(self, xgrid: np.ndarray, as_cdf: bool = False):

        """
        Returns the distribution of the hyper
        parameters mapped over a grid

        :param xgrid:
        :type xgrid: np.ndarray
        :param as_cdf:
        :type as_cdf: bool
        :returns:

        """
        y = np.zeros((self.n_samples, len(xgrid)))

        if self.alpha is None:

            dist = stats.norm

            generator = zip(self.mu, self.sigma)

            def out_func(func, x):

                for i, (mu, sigma) in enumerate(generator):

                    y[i, :] = func(x, loc=mu, scale=sigma)

        else:

            dist = stats.skewnorm

            generator = zip(self.mu, self.sigma, self.alpha)

            def out_func(func, x):

                for i, (mu, sigma, alpha) in enumerate(generator):

                    y[i, :] = func(x, loc=mu, scale=sigma, a=alpha)

        if as_cdf:

            func = dist.cdf

        else:

            func = dist.pdf

        out_func(func, xgrid)

        return y

    def plot_normal_population_distribution(
        self,
        ax: plt.Axes,
        hist_color: str,
        dist_color: str,
        n_bins: int,
        min_x,
        max_x,
        is_sim: bool,
    ):

        xgrid = np.linspace(min_x, max_x, 100)

        # if we have a simulation
        # then plot the data

        if is_sim:

            ax.hist(
                self.simulated_parameter,
                bins=n_bins,
                density=True,
                lw=2,
                fc=hist_color,
                ec=Colors.black,
                alpha=0.75,
            )

        y = self._distribution(xgrid)
        dist_plotter(xgrid, y, ax, alpha=0.75, color=dist_color, lw=0)

    def plot_normal_population_distribution_cdf(
        self,
        ax: plt.Axes,
        dist_color: str,
        object_color: str,
        is_sim: bool,
        min_x,
        max_x,
        object_lw=0.5,
        object_size=3,
    ):

        xgrid = np.linspace(min_x, max_x, 100)

        # extract the medians

        medians = np.array([np.median(x) for x in self.object_level])

        sort_idx = medians.argsort()

        qs: Quantile = extract_quantiles(self.object_level, level=90)

        dy = 1.0 / len(self.object_level)

        y = np.array([i * dy for i in range(len(self.object_level))])

        ax.hlines(
            y,
            qs.low[sort_idx],
            qs.high[sort_idx],
            color=object_color,
            zorder=5000,
            lw=object_lw,
        )

        # if we have a simulation
        # then plot the data

        if is_sim:

            ax.scatter(
                self.simulated_parameter[sort_idx],
                y,
                s=object_size,
                c=Colors.black,
            )

        # ax.plot(xgrid, stats.norm.pdf(xgrid, loc=, scale=0.5),  color="b")

        y = self._distribution(xgrid, as_cdf=True)

        dist_plotter(xgrid, y, ax, alpha=0.75, color=dist_color, lw=0)


@dataclass
class PosteriorContainer:
    flux: np.ndarray
    index: np.ndarray
    has_host_fit: bool
    host_nh: Optional[np.ndarray]
    log_nh_host_mu: Optional[ArrayLike]
    log_nh_host_sigma: Optional[ArrayLike]
    nh_host_alpha: Optional[ArrayLike]
    has_whim_fit: bool
    has_skew_fit: bool
    n0_whim: Optional[np.ndarray]
    t_whim: Optional[np.ndarray]
    index_mu: Optional[np.ndarray]
    index_sigma: Optional[np.ndarray]
    log_K_mu: Optional[np.ndarray]
    log_K_sigma: Optional[np.ndarray]
    has_whim_sim: bool
    has_host_sim: bool

    def to_hdf_group(self, hdf_grp: h5py.Group) -> None:

        """
        write the information to an HDF group

        :param hdf_grp:
        :type hdf_grp: h5py.Group
        :returns:

        """
        hdf_grp.attrs["has_host_fit"] = self.has_host_fit
        hdf_grp.attrs["has_whim_fit"] = self.has_whim_fit
        hdf_grp.attrs["has_host_sim"] = self.has_host_sim
        hdf_grp.attrs["has_whim_sim"] = self.has_whim_sim
        hdf_grp.attrs["has_skew_fit"] = self.has_skew_fit

        hdf_grp.create_dataset("flux", data=self.flux, compression="gzip")
        hdf_grp.create_dataset("index", data=self.index, compression="gzip")

        if self.has_host_fit:

            hdf_grp.create_dataset(
                "log_host_mu", data=self.log_nh_host_mu, compression="gzip"
            )
            hdf_grp.create_dataset(
                "log_host_sigma",
                data=self.log_nh_host_sigma,
                compression="gzip",
            )

            hdf_grp.create_dataset(
                "host_nh",
                data=self.host_nh,
                compression="gzip",
            )

        if self.has_skew_fit:

            hdf_grp.create_dataset(
                "nh_host_alpha", data=self.nh_host_alpha, compression="gzip"
            )

        if self.has_whim_fit:

            hdf_grp.create_dataset(
                "n0_whim", data=self.n0_whim, compression="gzip"
            )
            hdf_grp.create_dataset(
                "t_whim", data=self.t_whim, compression="gzip"
            )

        hdf_grp.create_dataset(
            "index_mu", data=self.index_mu, compression="gzip"
        )
        hdf_grp.create_dataset(
            "index_sigma", data=self.index_sigma, compression="gzip"
        )

        hdf_grp.create_dataset(
            "log_K_mu", data=self.log_K_mu, compression="gzip"
        )

        hdf_grp.create_dataset(
            "log_K_sigma", data=self.log_K_sigma, compression="gzip"
        )

    @classmethod
    def from_hdf_group(cls, hdf_grp: h5py.Group):

        n0_whim = None
        t_whim = None
        log_nh_host_mu = None
        log_nh_host_sigma = None
        host_nh = None
        nh_host_alpha = None

        if hdf_grp.attrs["has_host_fit"]:
            host_nh = hdf_grp["host_nh"][()]
            log_nh_host_mu = hdf_grp["log_host_mu"][()]
            log_nh_host_sigma = hdf_grp["log_host_sigma"][()]

        if hdf_grp.attrs["has_skew_fit"]:
            nh_host_alpha = hdf_grp["nh_host_alpha"][()]

        if hdf_grp.attrs["has_whim_fit"]:
            n0_whim = hdf_grp["n0_whim"][()]
            t_whim = hdf_grp["t_whim"][()]

        return cls(
            flux=hdf_grp["flux"][()],
            index=hdf_grp["index"][()],
            index_mu=hdf_grp["index_mu"][()],
            index_sigma=hdf_grp["index_sigma"][()],
            log_K_mu=hdf_grp["log_K_mu"][()],
            log_K_sigma=hdf_grp["log_K_sigma"][()],
            has_host_fit=hdf_grp.attrs["has_host_fit"],
            has_whim_fit=hdf_grp.attrs["has_whim_fit"],
            has_skew_fit=hdf_grp.attrs["has_skew_fit"],
            has_host_sim=hdf_grp.attrs["has_host_sim"],
            has_whim_sim=hdf_grp.attrs["has_whim_sim"],
            host_nh=host_nh,
            log_nh_host_mu=log_nh_host_mu,
            log_nh_host_sigma=log_nh_host_sigma,
            n0_whim=n0_whim,
            t_whim=t_whim,
            nh_host_alpha=nh_host_alpha,
        )

    @classmethod
    def from_stan_fit(
        cls, *stan_fit: av.InferenceData, has_host_sim: bool, has_whim_sim: bool
    ):
        """
        Create a fit object from a recent stan fit in
        memory

        :param cls:
        :type cls:
        :param inference_data:
        :type inference_data: av.InferenceData

        :returns:

        """

        if len(stan_fit) == 1:

            stan_posterior = stan_fit[0]

        else:

            stan_posterior = stan_fit[0]

            for idata in stan_fit[1:]:

                stan_posterior = av.concat(self._posterior, idata, dim="chain")

        n_grbs: int = stan_posterior.K.stack(
            sample=("chain", "draw")
        ).values.shape[0]

        flux: ArrayLike = stan_posterior.K.stack(
            sample=("chain", "draw")
        ).values
        index: ArrayLike = stan_posterior.index.stack(
            sample=("chain", "draw")
        ).values

        has_host_fit: bool = False
        host_nh: Optional[ArrayLike] = None
        log_nh_host_mu: Optional[ArrayLike] = None
        log_nh_host_sigma: Optional[ArrayLike] = None

        nh_host_alpha = None

        try:

            nh_host_alpha = stan_posterior.host_alpha.stack(
                sample=("chain", "draw")
            ).values

            has_skew_fit = True

        except AttributeError:

            has_skew_fit = False

        try:
            host_nh = stan_posterior.nH_host_norm.stack(
                sample=("chain", "draw")
            ).values

            log_nh_host_mu = stan_posterior.log_nH_host_mu_raw.stack(
                sample=("chain", "draw")
            ).values

            log_nh_host_sigma = stan_posterior.log_nH_host_sigma.stack(
                sample=("chain", "draw")
            ).values

            has_host_fit = True

        except AttributeError:

            # we do not have a host gas fit

            pass

        has_whim_fit: bool = False
        n0_whim = None
        t_whim = None

        try:
            n0_whim = stan_posterior.n0_whim.stack(
                sample=("chain", "draw")
            ).values
            t_whim = stan_posterior.t_whim.stack(
                sample=("chain", "draw")
            ).values
            has_whim_fit = True

        except AttributeError:

            # we do not have a whim fit

            pass

        # group properties

        index_mu: ArrayLike = stan_posterior.index_mu.stack(
            sample=("chain", "draw")
        ).values

        index_sigma: ArrayLike = stan_posterior.index_sigma.stack(
            sample=("chain", "draw")
        ).values

        log_K_mu: ArrayLike = stan_posterior.log_K_mu.stack(
            sample=("chain", "draw")
        ).values

        log_K_sigma: ArrayLike = stan_posterior.log_K_sigma.stack(
            sample=("chain", "draw")
        ).values

        return cls(
            flux=flux,
            index=index,
            has_host_fit=has_host_fit,
            has_skew_fit=has_skew_fit,
            host_nh=host_nh,
            log_nh_host_mu=log_nh_host_mu,
            log_nh_host_sigma=log_nh_host_sigma,
            nh_host_alpha=nh_host_alpha,
            has_whim_fit=has_whim_fit,
            n0_whim=n0_whim,
            t_whim=t_whim,
            index_mu=index_mu,
            index_sigma=index_sigma,
            log_K_mu=log_K_mu,
            log_K_sigma=log_K_sigma,
            has_whim_sim=has_whim_sim,
            has_host_sim=has_host_sim,
        )


class Fit:
    def __init__(
        self, database: Database, posterior: PosteriorContainer, model_name: str
    ):

        """
        Holds and displays the fit to the data as well as holds the data

        :param database:
        :type database: Database
        :param posterior:
        :type posterior: PosteriorContainer
        :param model_name:
        :type model_name: str
        :returns:

        """
        self._model_name: str = model_name

        # this simply attaches every thing to the class

        self._catalog: XRTCatalog = database.catalog
        self._posterior: PosteriorContainer = posterior
        self._database: Database = database

        self._flux: ArrayLike = self._posterior.flux
        self._index: ArrayLike = self._posterior.index

        self._has_host_fit: bool = self._posterior.has_host_fit
        self._host_nh: Optional[ArrayLike] = self._posterior.host_nh
        self._log_nh_host_mu: Optional[
            ArrayLike
        ] = self._posterior.log_nh_host_mu
        self._log_nh_host_sigma: Optional[
            ArrayLike
        ] = self._posterior.log_nh_host_sigma

        self._nh_host_alpha = self._posterior.nh_host_alpha

        self._host_nh = self._posterior.host_nh

        self._log_nh_host_mu = self._posterior.log_nh_host_mu
        self._log_nh_host_sigma = self._posterior.log_nh_host_sigma

        self._has_host_fit = self._posterior.has_host_fit

        self._has_whim_fit: bool = self._posterior.has_whim_fit

        self._n0_whim = self._posterior.n0_whim
        self._t_whim = self._posterior.t_whim

        # group properties

        self._index_mu: np.ndarray = self._posterior.index_mu

        self._index_sigma: ArrayLike = self._posterior.index_sigma

        self._log_K_mu: np.ndarray = self._posterior.log_K_mu

        self._log_K_sigma: ArrayLike = self._posterior.log_K_sigma

        self._grbs = self._catalog.grbs

        self._n_grbs = len(self._grbs)

        self._n_samples: int = len(self._index[0])

        self._is_sim = self._catalog.is_sim

        # Set up hyper distributions

        self._hyper_index = HyperObject(
            self._index_mu,
            self._index_sigma,
            self._index,
            self._n_samples,
            simulated_parameter=self._catalog.index_sim,
        )

        if self._is_sim:

            tmp = np.log10(self._catalog.flux_sim)
            tmp2 = np.log10(self._catalog.nH_host_sim) + 22

        else:

            tmp = None
            tmp2 = None

        self._hyper_flux = HyperObject(
            self._log_K_mu,
            self._log_K_sigma,
            np.log10(self._flux),
            self._n_samples,
            simulated_parameter=tmp,
        )

        self._hyper_nh = HyperObject(
            self._log_nh_host_mu + 22,
            self._log_nh_host_sigma,
            np.log10(self._host_nh * 1e22),
            self._n_samples,
            alpha=self._nh_host_alpha,
            simulated_parameter=tmp2,
        )

        # if it is a sim, lets go ahead
        # and see what's in there

        self._has_whim_sim: bool = False
        self._has_host_sim: bool = False

        # some special properties for the simulation

        if self._catalog.is_sim:

            if self._catalog.n0_sim is not None:

                self._has_whim_sim = True

            if self._catalog.nH_host_sim is not None:

                self._has_host_sim = True

    @classmethod
    def from_live_fit(
        cls, *stan_fit: av.InferenceData, database: Database, model_name
    ) -> "Fit":
        """
        Create a fit object from a recent stan fit in
        memory

        :param cls:
        :type cls:
        :param inference_data:
        :type inference_data: av.InferenceData
        :param database:
        :type database: Database
        :returns:

        """

        # if it is a sim, lets go ahead
        # and see what's in there

        has_whim_sim: bool = False
        has_host_sim: bool = False

        if database.catalog.is_sim:

            if database.catalog.n0_sim is not None:

                has_whim_sim = True

            if database.catalog.nH_host_sim is not None:

                has_host_sim = True

        posterior = PosteriorContainer.from_stan_fit(
            *stan_fit,
            has_host_sim=has_host_sim,
            has_whim_sim=has_whim_sim,
        )

        return cls(
            database=database, posterior=posterior, model_name=model_name
        )

    @classmethod
    def from_file(cls, file_name: str) -> "Fit":

        """
        read the fit and database from a previous save

        :param cls:
        :type cls:
        :param file_name:
        :type file_name: str
        :returns:

        """
        with h5py.File(file_name, "r") as f:

            database: Database = Database.read(f["database"])

            posterior: PosteriorContainer = PosteriorContainer.from_hdf_group(
                f["posterior"]
            )

            model_name = f.attrs["model_name"]

        return cls(
            database=database, posterior=posterior, model_name=model_name
        )

    def write(self, file_name: str) -> None:

        out = Path(file_name)

        with h5py.File(str(out), "w") as f:

            f.attrs["model_name"] = self._model_name

            db_grp = f.create_group("database")

            self._database.write(db_grp)

            fit_grp = f.create_group("posterior")

            self._posterior.to_hdf_group(fit_grp)

    @property
    def catalog(self) -> XRTCatalog:
        """
        The catalog of the GRBs

        :returns:

        """
        return self._catalog

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
    def host_alpha(self) -> Optional[np.ndarray]:

        return self._nh_host_alpha

    @property
    def n0_whim(self) -> ArrayLike:
        return self._n0_whim

    @property
    def t_whim(self) -> ArrayLike:
        return self._t_whim

    @property
    def index_mu(self) -> ArrayLike:
        return self._index_mu

    @property
    def index_sigma(self) -> ArrayLike:
        return self._index_sigma

    @property
    def log_K_mu(self) -> ArrayLike:
        return self._log_K_mu

    @property
    def log_K_sigma(self) -> ArrayLike:
        return self._log_K_sigma

    @property
    def log_nh_host_mu(self) -> np.ndarray:
        return self._log_nh_host_mu

    @property
    def log_nh_host_sigma(self) -> ArrayLike:
        return self._log_nh_host_sigma

    def plot_nh_host_distribution(
        self,
        hist_color: str = Colors.grey,
        dist_color: str = Colors.green,
        object_color: str = Colors.purple,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 10,
        as_cdf: bool = False,
        object_lw: float = 0.5,
        object_size: float = 3,
    ) -> plt.Figure:

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        xgrid = np.linspace(19.0, 25, 100)

        if as_cdf:

            self._hyper_nh.plot_normal_population_distribution_cdf(
                ax,
                dist_color,
                object_color,
                self._catalog.is_sim,
                19,
                25,
                object_lw,
                object_size,
            )

        else:

            self._hyper_nh.plot_normal_population_distribution(
                ax,
                hist_color,
                dist_color,
                n_bins,
                19,
                25,
                self._catalog.is_sim,
            )

        ax.set_xlabel("log10(nH host)")

        return fig

    def plot_index_distribution(
        self,
        hist_color: str = Colors.grey,
        dist_color: str = Colors.green,
        object_color: str = Colors.purple,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 10,
        as_cdf: bool = False,
        object_lw: float = 0.5,
        object_size: float = 3,
    ) -> plt.Figure:

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        if as_cdf:

            self._hyper_index.plot_normal_population_distribution_cdf(
                ax,
                dist_color,
                object_color,
                self._catalog.is_sim,
                -4,
                0,
                object_lw,
                object_size,
            )

        else:

            self._hyper_index.plot_normal_population_distribution(
                ax,
                hist_color,
                dist_color,
                n_bins,
                -4,
                0,
                self._catalog.is_sim,
            )

        ax.set_xlabel("spectral index")

        return fig

    def plot_flux_distribution(
        self,
        hist_color: str = Colors.grey,
        dist_color: str = Colors.green,
        object_color: str = Colors.purple,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 10,
        as_cdf: bool = False,
        object_lw: float = 0.5,
        object_size: float = 3,
    ) -> plt.Figure:

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        if as_cdf:

            self._hyper_flux.plot_normal_population_distribution_cdf(
                ax,
                dist_color,
                object_color,
                self._catalog.is_sim,
                -12,
                -8,
                object_lw,
                object_size,
            )

        else:

            self._hyper_flux.plot_normal_population_distribution(
                ax,
                hist_color,
                dist_color,
                n_bins,
                -12,
                -8,
                self._catalog.is_sim,
            )

        ax.set_xlabel("flux distribution")

        return fig

    def _get_spectrum(self, id: int) -> ModelContainer:
        """


        :param id:
        :type id:
        :returns:

        """

        with_whim: bool = False
        with_host: bool = False

        if self._has_whim_fit or self._has_whim_sim:

            with_whim = True

        if self._has_host_sim or self._has_host_fit:

            with_host = True

        # now get the sepctrum from the catalog

        grb: str = self._catalog.grbs[id]

        return self._catalog.catalog[grb].get_spectrum(
            with_host=with_host, with_whim=with_whim
        )

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

        if show_truth and self._catalog.is_sim:

            ax.loglog(
                self._catalog.z,
                1e22 * self._catalog.nH_host_sim,
                "o",
                color=Colors.black,
                markersize=3.0,
                alpha=1.0,
                zorder=-1000,
            )

        qs: Quantile = extract_quantiles(1e22 * self._host_nh, level=90)

        width = rank_quantile_width(qs)

        my_cmap = build_custom_continuous_cmap(Colors.green, Colors.purple)

        colors = array_to_cmap(width, cmap=my_cmap, use_log=True)

        for i in range(self._n_grbs):

            ax.vlines(
                self._catalog.z[i],
                qs.low[i],
                qs.high[i],
                color=colors[i],
                linewidth=0.7,
            )

        ax.set_ylabel(r"host nH (cm$^{-2}$)")

        ax.set_xlabel("z")

        ax.set_xscale("log")

        ax.set_yscale("log")

        ax.set_xlim(right=10)

        return fig

    def get_plugin_and_model(self, id: int) -> Tuple[OGIPLike, Model]:

        plugin: OGIPLike = self._database.plugins[self._grbs[id]]

        plugin.integration_method = "riemann"

        # if self._data_path is not None:

        #     bpath = self._data_path / f"grb{self._grbs[id]}/"

        # else:

        #     return

        # plugin = OGIPLike(
        #     "xrt",
        #     observation=bpath / "apc.pha",
        #     background=bpath / "apc_bak.pha",
        #     response=bpath / "apc.rsp",
        #     spectrum_number=1,
        # )

        # get the model container object
        model_container: ModelContainer = self._get_spectrum(id)

        if self._has_whim_fit:

            model = model_container.model_all

        elif self._has_host_fit:

            model = model_container.model_host

        return plugin, model

    def _extract_samples(self, id: int) -> np.ndarray:

        """
        extract the samples for a GRB for a fit

        :param id:
        :type id: int
        :returns:

        """
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

        else:

            samples = np.vstack((self._flux[id], self._index[id]))

        return samples

    def plot_ppc(
        self, id: int, n_sims: int = 100, min_rate: Optional[float] = None
    ) -> plt.Figure:

        samples = self._extract_samples(id)

        analysis = self._database.build_3ml_analysis(
            id, with_whim=self._has_whim_fit
        )

        _, model = self.get_plugin_and_model(id)

        with np.errstate(invalid="ignore"):
            ppc = compute_postpc(
                analysis=analysis,
                n_sims=n_sims,
                overwrite=True,
                return_ppc=True,
                direct_samples=samples.T,
                direct_model=model,
            )

        fig = ppc.grb.plot(
            bkg_subtract=True,
            levels=[95, 68],
            colors=[Colors.purple, Colors.dark_purple],
            lc=Colors.green,
            min_rate=min_rate,
        )

        ax = fig.get_axes()[0]

        ax.set_xscale("linear")

        return fig

    def plot_data_spectrum(
        self,
        id: int,
        min_rate: float = -99,
        model_color=Colors.green,
        data_color=Colors.purple,
        thin=2,
    ) -> plt.Figure:

        o, model = self.get_plugin_and_model(id)

        if o is None:

            return

        samples = self._extract_samples(id)
        with np.errstate(invalid="ignore"):

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
                source_only=True,
            )

        ax = fig.get_axes()[0]

        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.set_xlim(0.3)

        return fig

    def plot_model_spectrum(
        self,
        id: int,
        thin: int = 2,
        show_sim: bool = True,
        show_legend: bool = True,
    ) -> plt.Figure:

        """
        plot the spectrum of the corresponding GRB in photo
        space

        :param id:
        :type id: int
        :param thin:
        :type thin:
        :param show_sim:
        :param show_legend:
        :returns:

        """
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
            custom_lines.append(Line2D([0], [0], color=Colors.purple, lw=2))
        if self._has_host_fit:
            labels.append("Host posterior")
            custom_lines.append(Line2D([0], [0], color=Colors.green, lw=2))

        samples = self._extract_samples(id)

        # scroll through the samples

        y = np.empty((3, len(samples.T[::thin]), len(ene)))

        with np.errstate(invalid="ignore"):

            for i, sample in enumerate(samples.T[::thin]):

                if self._has_whim_fit:

                    model_container.model_all.set_free_parameters(sample)

                    y[
                        2, i, :
                    ] = model_container.model_all.get_point_source_fluxes(
                        0, ene
                    )

                if self._has_host_fit:

                    model_container.model_mw.set_free_parameters(sample[:2])

                    model_container.model_host.set_free_parameters(sample[:3])

                    y[
                        1, i, :
                    ] = model_container.model_host.get_point_source_fluxes(
                        0, ene
                    )

                model_container.model_pl.set_free_parameters(sample[:2])

                y[0, i, :] = model_container.model_pl.get_point_source_fluxes(
                    0, ene
                )

        # plotting

        dist_plotter(ene, y[0], ax, color=Colors.yellow, alpha=0.5)

        if self._has_host_fit:

            dist_plotter(ene, y[1], ax, color=Colors.green, alpha=0.5)

        if self._has_whim_fit:

            dist_plotter(ene, y[2], ax, color=Colors.purple, alpha=0.5)

        if self._catalog.is_sim and show_sim:

            simulated_parameters = card.simulated_parameters

            if (model_container.model_all is not None) and self._has_whim_sim:

                # ok, we have some whim
                labels.append("Total Simualted")
                custom_lines.append(Line2D([0], [0], color=Colors.black, lw=2))
                model_container.model_all.set_free_parameters(
                    simulated_parameters
                )

                ax.loglog(
                    ene,
                    model_container.model_all.get_point_source_fluxes(0, ene),
                    color=Colors.light_grey,
                    lw=0.5,
                )

            if model_container.model_host is not None:

                # ok, we have some host gas (MW as well)
                labels.append("Simulation Host included")
                custom_lines.append(Line2D([0], [0], color=Colors.grey, lw=2))
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
                    color=Colors.grey,
                    lw=0.5,
                )

            model_container.model_pl.set_free_parameters(
                simulated_parameters[:2]
            )

            ax.loglog(
                ene,
                model_container.model_pl.get_point_source_fluxes(0, ene),
                color=Colors.black,
                lw=0.5,
            )

        if show_legend:

            ax.legend(custom_lines, labels)

        ax.set_xlabel("energy (keV)")
        ax.set_ylabel(r"flux phts s$^{-1}$kev$^{-1}$cm$^{-2}$)")

        return fig
