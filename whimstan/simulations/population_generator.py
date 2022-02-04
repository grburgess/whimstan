from typing import Optional

import astropy.units as u
import numpy as np
import popsynth as ps
import scipy.special as sf
from astromodels import Model, PointSource, Powerlaw_Eflux, TbAbs
from astropy.coordinates import SkyCoord
from bb_astromodels import Integrate_Absori
from gdpyc import GasMap
from popsynth.utils.progress_bar import progress_bar
from threeML.plugins.OGIPLike import OGIPLike

from .cloud import Cloud


class SchechterSampler(ps.AuxiliarySampler):

    _auxiliary_sampler_name = "SchechterSampler"

    Lmin = ps.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    alpha = ps.auxiliary_sampler.AuxiliaryParameter(default=1)

    def __init__(self, name="plaw_flux"):
        """
        A Schechter luminosity function
        :param name: the name
        :returns: None
        :rtype:
        """

        super(SchechterSampler, self).__init__(name=name, observed=False)

    def true_sampler(self, size):
        """FIXME! briefly describe function
        :param size:
        :returns:
        :rtype:
        """

        xs = 1 - np.random.uniform(size=size)
        self._true_values = self.Lmin * sf.gammaincinv(1 + self.alpha, xs)


class HostGas(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "HostGas"

    nh_mean = ps.auxiliary_sampler.AuxiliaryParameter(default=22, vmin=0)
    zratio = ps.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0, vmax=1)

    def __init__(self):
        """
        The value of the host gas density in 1/cm2
        """

        # pass up to the super class

        super(HostGas, self).__init__("host_nh", observed=False)

    def true_sampler(self, size):

        # create a cloud with this simulation
        cloud = Cloud(zr_ratio=self.zratio)

        # sample the latent values for this property

        path_length = np.array([cloud.sample() for i in range(size)])

        self._true_values = np.power(10.0, self.nh_mean) * path_length


class HostGasVari(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "HostGasVari"

    def __init__(self):
        """
        The value of the host gas density in 1/cm2
        """

        # pass up to the super class

        super(HostGasVari, self).__init__("host_nh", observed=False)

    def true_sampler(self, size):

        zr_ratios = self.secondary_samplers["zr_ratio"]
        nh_local = self.secondary_samplers["nh_local"]

        # create a cloud with this simulation
        clouds = [Cloud(zr_ratio=zr) for zr in zr_ratios.true_values]

        # sample the latent values for this property

        path_length = np.array([cloud.sample() for cloud in clouds])

        self._true_values = np.power(10.0, nh_local.true_values) * path_length


class ZRSampler(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "ZRSampler"
    zmin = ps.AuxiliaryParameter(vmin=-99, vmax=0)

    def __init__(self):
        """
        Log uniform sampler for the Z/R ratio
        """

        # pass up to the super class

        super(ZRSampler, self).__init__("zr_ratio", observed=False)

    def true_sampler(self, size):

        self._true_values = 10.0 ** np.random.uniform(self.zmin, 0, size=size)


class MilkyWayGas(ps.AuxiliarySampler):
    _auxiliary_sampler_name = "MilkyWayGas"

    def __init__(self):
        """
        The value of the milky way gas density in 1/cm2
        """

        # pass up to the super class

        super(MilkyWayGas, self).__init__(
            "mw_nh", observed=False, uses_sky_position=True
        )

    def true_sampler(self, size):

        mw_nh = np.empty(size)

        coord = SkyCoord(ra=self._ra, dec=self._dec, unit="deg", frame="icrs")

        for i, c in enumerate(coord):

            mw_nh[i] = GasMap.nhf(c, nhmap="DL", radius=1 * u.deg).value

        self._true_values = mw_nh


class MWGasSelection(ps.SelectionProbability):

    _selection_name = "MWGasSelection"

    gas_limit = ps.SelectionParameter(vmin=0)

    def __init__(self, name="mw gas selector"):
        """
        places a limit on the amount of MW gas allowed for
        each object in the sample

        """

        super(MWGasSelection, self).__init__(name=name, use_obs_value=True)

    def draw(self, size: int):

        self._selection = self._observed_value < np.power(10.0, self.gas_limit)


class CountsSelector(ps.SelectionProbability):

    _selection_name = "CountsSelector"

    counts_limit = ps.SelectionParameter(vmin=0)

    def __init__(self, name="counts selector"):
        """
        places a min limit on the number of counts
        """

        super(CountsSelector, self).__init__(name=name, use_obs_value=True)

    def draw(self, size: int):

        self._selection = self._observed_value > self.counts_limit


class GalacticPlaceDistanceSelection(ps.SpatialSelection):

    _selection_name = "GalacticPlaceDistanceSelection"

    b_limit = ps.SelectionParameter(vmin=0, vmax=90)
    z_min = ps.SelectionParameter(vmin=0)

    def __init__(self, name="mw plane selector"):
        """
        places a limit above the galactic plane for objects
        """
        super().__init__(name=name)

    def draw(self, size: int):

        g_coor = SkyCoord(
            self._spatial_distribution.ra,
            self._spatial_distribution.dec,
            unit="deg",
            frame="icrs",
        ).transform_to("galactic")

        selection = (g_coor.b.deg >= self.b_limit) | (
            g_coor.b.deg <= -self.b_limit
        )

        selection2 = self._spatial_distribution.distances > self.z_min

        self._selection = selection & selection2


class ExposureSampler(ps.AuxiliarySampler):

    _auxiliary_sampler_name = "ExposureSampler"

    def __init__(self, low, high):

        self._low = low
        self._high = high

        super().__init__(name="exposure", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.uniform(self._low, self._high, size)


class ObscuredFluxSampler(ps.DerivedLumAuxSampler):

    _auxiliary_sampler_name = "ObscuredFluxSampler"

    def __init__(
        self,
        a: float = 0.4,
        b: float = 10,
        whim_n0: Optional[float] = None,
        whim_T: Optional[float] = None,
        use_mw_gas: bool = True,
        use_host_gas: bool = True,
        plugin_for_counts: Optional[OGIPLike] = None,
    ):
        """
        computes the obscured flux from the GRB by integrating the spectra


        :param a: lower bound of integration
        :param b: upper bound of integration
        :param whim_n0: the optional whim particle density
        :param whim_T: the optional whim temperature in K
        """

        self._a = a
        self._b = b

        self._whim_T = whim_T
        self._whim_n0 = whim_n0

        self._use_mw_gas = use_mw_gas
        self._use_host_gas = use_host_gas

        self._plugin_for_counts: Optional[OGIPLike] = plugin_for_counts

        if self._plugin_for_counts is not None:

            self._plugin_for_counts.model_integrate_method = "riemann"

        super(ObscuredFluxSampler, self).__init__(
            "obscured_flux", uses_distance=True
        )

    def true_sampler(self, size):

        # kev to erg
        kev2erg = 1.6021766339999998e-09

        n_energies_for_intergration = 50

        intergration_energies = np.geomspace(
            self._a, self._b, num=n_energies_for_intergration
        )

        fluxes = np.empty(size)
        counts = np.empty(size)

        # we want to have the flux measured in the XRT so
        # we need to integrate the obscured flux

        for i in progress_bar(range(size), desc="computing obscured fluxes"):

            spec = Powerlaw_Eflux(
                F=self._secondary_samplers["plaw_flux"].true_values[i]
                / (4 * np.pi * (self.luminosity_distance[i] ** 2)),
                index=self._secondary_samplers["spec_idx"].true_values[i],
                a=self._a,
                b=self._b,
            )
            if self._use_mw_gas:
                spec *= TbAbs(
                    NH=self._secondary_samplers["mw_nh"].true_values[i]
                    / (1.0e22),
                    redshift=0,
                )
            if self._use_host_gas:

                tmp = TbAbs(redshift=self._distance[i])

                tmp.NH.bounds = (None, None)

                tmp.NH = self._secondary_samplers["host_nh"].true_values[i] / (
                    1.0e22
                )

                spec *= tmp
            # add on the WHIM if needed
            if (self._whim_n0 is not None) and (self._whim_T is not None):

                spec = spec * Integrate_Absori(
                    n0=self._whim_n0,
                    temp=self._whim_T,
                    redshift=self._distance[i],
                )

            # now compute the energy integral.
            # using the slower quad here because
            # the gas models have a shit load of lines

            # flux = quad(lambda x: x * spec(x), self._a, self._b)[0] * kev2erg

            if self._plugin_for_counts is not None:

                # self._plugin_for_counts._background_spectrum._exposure = new_exposure
                # self._plugin_for_counts._background_spectrum._exposure = new_exposure

                ps = PointSource("tmp", 0, 0, spectral_shape=spec)

                model = Model(ps)

                self._plugin_for_counts.set_model(model)

                self._plugin_for_counts.set_active_measurements("0.3-10")

                count_spectrum = (
                    self._plugin_for_counts._evaluate_model()
                    * self._secondary_samplers["exposure"].true_values[i]
                )

                total_counts = (
                    count_spectrum[self._plugin_for_counts.mask]
                ).sum()

                counts[i] = total_counts

            flux = (
                np.trapz(
                    intergration_energies * spec(intergration_energies),
                    intergration_energies,
                )
                * kev2erg
            )

            fluxes[i] = flux

        if self._plugin_for_counts is None:

            self._true_values = fluxes

        else:

            self._true_values = counts

        self._fluxes = fluxes

    def compute_luminosity(self):

        # have to compute back to a luminosity

        return (4.0 * np.pi * self.luminosity_distance ** 2) * self._fluxes


def create_population(
    r0: float = 5,
    a: float = 0.0157,
    rise: float = 0.118,
    decay: float = 4.2,
    peak: float = 3.4,
    z_max: float = 10.0,
    Lmin: float = 1e46,
    alpha: float = 1.5,
    host_gas_mean: float = 23,
    host_gas_sigma: float = 0.5,
    host_gas_cloud_ratio: float = 0.1,
    mw_nh_limit: Optional[float] = None,
    b_limit: Optional[float] = None,
    use_clouds: bool = True,
    vari_clouds: bool = True,
    spec_idx_mean: float = -2.0,
    spec_idx_std: float = 0.2,
    use_mw_gas: bool = True,
    use_host_gas: bool = True,
    whim_n0: Optional[float] = None,
    whim_T: Optional[float] = None,
    demo_plugin: Optional[OGIPLike] = None,
    counts_limit: Optional[float] = None,
    exposure_high: Optional[float] = None,
    exposure_low: Optional[float] = None,
    z_min: float = 0,
) -> ps.PopulationSynth:

    """
    create a population syn

    :param r0:
    :type r0: float
    :param a:
    :type a: float
    :param rise:
    :type rise: float
    :param decay:
    :type decay: float
    :param peak:
    :type peak: float
    :param z_max:
    :type z_max: float
    :param Lmin:
    :type Lmin: float
    :param alpha:
    :type alpha: float
    :param host_gas_mean:
    :type host_gas_mean: float
    :param host_gas_sigma:
    :type host_gas_sigma: float
    :param host_gas_cloud_ratio:
    :type host_gas_cloud_ratio: float
    :param mw_nh_limit:
    :type mw_nh_limit: Optional[float]
    :param b_limit:
    :type b_limit: Optional[float]
    :param use_clouds:
    :type use_clouds: bool
    :param vari_clouds:
    :type vari_clouds: bool
    :param spec_idx_mean:
    :type spec_idx_mean: float
    :param spec_idx_std:
    :type spec_idx_std: float
    :param use_mw_gas:
    :type use_mw_gas: bool
    :param use_host_gas:
    :type use_host_gas: bool
    :param whim_n0:
    :type whim_n0: Optional[float]
    :param whim_T:
    :type whim_T: Optional[float]
    :param demo_plugin:
    :type demo_plugin: Optional[OGIPLike]
    :param counts_limit:
    :type counts_limit: Optional[float]
    :param exposure_high:
    :type exposure_high: Optional[float]
    :param exposure_low:
    :type exposure_low: Optional[float]
    :param z_min:
    :type z_min: float
    :returns:

    """
    if use_host_gas:
        if use_clouds:

            # the host galaxy gas will be
            # created by embedding GRBs in
            # clouds

            if vari_clouds:

                host_nh = HostGasVari()

                zr_sampler = ZRSampler()

                zr_sampler.zmin = np.log10(host_gas_cloud_ratio)

                nh_local = ps.TruncatedNormalAuxSampler(name="nh_local")
                nh_local.mu = host_gas_mean
                nh_local.sigma = host_gas_sigma
                nh_local.lower = 20
                nh_local.upper = 26

                host_nh.set_secondary_sampler(nh_local)
                host_nh.set_secondary_sampler(zr_sampler)

            else:

                host_nh = HostGas()
                host_nh.nh_mean = host_gas_mean
                host_nh.zratio = host_gas_cloud_ratio

        else:

            # the host gas will be drawn from
            # a log normal which is empirical

            host_nh = ps.aux_samplers.Log10NormalAuxSampler(
                name="host_nh", observed=False
            )
            host_nh.mu = host_gas_mean
            host_nh.sigma = 0.5

    if use_mw_gas:
        # there are no random variables for the milky way gas
        mw_nh = MilkyWayGas()

        if mw_nh_limit is not None:

            mws = MWGasSelection()

            mws.gas_limit = mw_nh_limit

            mw_nh.set_selection_probability(mws)

    # GRB spectrum

    # sample the spectral index
    # of the power law

    spec_idx = ps.aux_samplers.NormalAuxSampler(name="spec_idx", observed=False)
    spec_idx.mu = spec_idx_mean
    spec_idx.sigma = spec_idx_std

    # sample the the intergral energy flux of
    # the power law
    powerlaw = SchechterSampler()
    powerlaw.Lmin = Lmin
    powerlaw.alpha = alpha

    # now we compute the "obscured" luminosity
    # that would lead to the flux actually observed
    # by the XRT

    # if we include a plugin here, then the selection is on the counts

    ls = ObscuredFluxSampler(
        whim_n0=whim_n0,
        whim_T=whim_T,
        use_mw_gas=use_mw_gas,
        use_host_gas=use_host_gas,
        plugin_for_counts=demo_plugin,
    )

    ls.set_secondary_sampler(spec_idx)
    ls.set_secondary_sampler(powerlaw)
    if use_host_gas:
        ls.set_secondary_sampler(host_nh)
    if use_mw_gas:
        ls.set_secondary_sampler(mw_nh)

    if demo_plugin is not None:

        exposure = ExposureSampler(low=exposure_low, high=exposure_high)

        ls.set_secondary_sampler(exposure)

        count_selector = CountsSelector()

        count_selector.counts_limit = counts_limit

        ls.set_selection_probability(count_selector)

    pop_gen: ps.PopulationSynth = ps.populations.SFRPopulation(
        r0=r0,
        a=a,
        rise=rise,
        decay=decay,
        peak=peak,
        r_max=z_max,
    )

    pop_gen.add_observed_quantity(ls)

    if b_limit is not None:

        gps = GalacticPlaceDistanceSelection()
        gps.b_limit = b_limit
        gps.z_min = z_min

        pop_gen.add_spatial_selector(gps)

    return pop_gen
