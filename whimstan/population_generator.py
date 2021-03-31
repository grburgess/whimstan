from pathlib import Path
from typing import Dict, List, Optional

import numba as nb
import numpy as np
import popsynth
import scipy.special as sf
from astromodels import Powerlaw_Eflux, TbAbs
from astropy.coordinates import SkyCoord
import astropy.units as u
from bb_astromodels import Integrate_Absori
from gdpyc import GasMap
from popsynth.utils.progress_bar import progress_bar
from scipy.integrate import quad


class Cloud(object):
    def __init__(self, R=1, zr_ratio=1.):

        self._R = R
        self._R2 = R * R
        self._zr_ratio = zr_ratio
        self._Z = self._R * self._zr_ratio
        self._Z2 = self._Z * self._Z
        self._size_vec = np.array([self._R, self._R, self._Z])
        self._size_vec2 = self._size_vec**2

    def sample(self):

        p = self.generate_point_inside()
        u = self.get_unit_vector()

        pl = self.compute_path_length(p, u)

        return pl

    def generate_point_inside(self):

        flag = True
        while flag:
            p = np.array([np.random.uniform(-self._R, self._R),
                          np.random.uniform(-self._R, self._R),
                          np.random.uniform(-self._Z, self._Z)])

            test_val = (p**2).dot(1./self._size_vec2)

            if test_val <= 1:

                flag = False

        return p

    def get_unit_vector(self):

        v = np.random.normal(size=3)
        u = v / np.linalg.norm(v)

        return u

    def compute_path_length(self, p, u):

        b = 2 * (p * u).dot(1./self._size_vec2)
        a = (u**2).dot(1./self._size_vec2)
        c = (p**2).dot(1./self._size_vec2)

        l = (-b + np.sqrt(b*b - 4 * a * (c-1)))/(2*a)

        return l


class SchechterSampler(popsynth.AuxiliarySampler):

    Lmin = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    alpha = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1)

    def __init__(self, name="plaw_flux"):
        """
        A Schechter luminosity function
        :param name: the name
        :returns: None
        :rtype:
        """

        super(SchechterSampler, self).__init__(name=name,
                                               observed=False)

    def true_sampler(self, size):
        """FIXME! briefly describe function
        :param size:
        :returns:
        :rtype:
        """

        xs = 1 - np.random.uniform(size=size)
        self._true_values = self.Lmin * sf.gammaincinv(1 + self.alpha, xs)


class HostGas(popsynth.AuxiliarySampler):

    nh_mean = popsynth.auxiliary_sampler.AuxiliaryParameter(default=22, vmin=0)
    zratio = popsynth.auxiliary_sampler.AuxiliaryParameter(
        default=1, vmin=0, vmax=1)

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

        self._true_values = np.power(10., self.nh_mean) * path_length


class MilkyWayGas(popsynth.AuxiliarySampler):

    def __init__(self):
        """
        The value of the milky way gas density in 1/cm2
        """

        # pass up to the super class

        super(MilkyWayGas, self).__init__(
            "mw_nh", observed=False, uses_sky_position=True)

    def true_sampler(self, size):

        mw_nh = np.empty(size)

        for i, (ra, dec) in enumerate(zip(self._ra, self._dec)):

            coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

            mw_nh[i] = GasMap.nhf(coord, nhmap='DL', radius=1*u.deg).value

        self._true_values = mw_nh


class ObscuredFluxSampler(popsynth.DerivedLumAuxSampler):

    def __init__(self, a: float = .4, b: float = 15, whim_n0: Optional[float] = None, whim_T: Optional[float] = None):
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

        super(ObscuredFluxSampler, self).__init__(
            "obscured_flux", uses_distance=True)

    def true_sampler(self, size):

        # kev to erg
        kev2erg = 1.6021766339999998e-09

        out = np.empty(size)

        # we want to have the flux measured in the XRT so
        # we need to integrate the obscured flux

        for i in progress_bar(range(size), desc="computing obscured fluxes"):

            spec = Powerlaw_Eflux(F=self._secondary_samplers["plaw_flux"].true_values[i]/(4 * np.pi * (self.luminosity_distance[i]**2)),
                                  index=self._secondary_samplers["spec_idx"].true_values[i],
                                  a=self._a,
                                  b=self._b) * TbAbs(NH=self._secondary_samplers["mw_nh"].true_values[i]/(1.e22),
                                                     redshift=0) * TbAbs(NH=self._secondary_samplers["host_nh"].true_values[i]/1.E22,
                                                                         redshift=self._distance[i])

            # add on the WHIM if needed
            if (self._whim_n0 is not None) and (self._whim_T is not None):

                spec = spec * \
                    Integrate_Absori(n0=self._whim_n0,
                                     temp=self._whim_T, redshift=self._distance[i])

            # now compute the energy integral.
            # using the slower quad here because
            # the gas models have a shit load of lines
            flux = quad(lambda x: x * spec(x), self._a, self._b)[0] * kev2erg

            out[i] = flux

        self._true_values = out

    def compute_luminosity(self):

        # have to compute back to a luminosity

        return (4.0 * np.pi * self.luminosity_distance**2) * self._true_values


def create_simulation(r0: float = 5,
                      rise: float = 1.,
                      decay: float = 4.0,
                      peak: float = 1.5,
                      Lmin: float = 1e46,
                      alpha: float = 1.5,
                      host_gas_mean: float = 23,
                      host_gas_cloud_ratio: float = 0.1,
                      use_clouds: bool = True,
                      spec_idx_mean: float = -2.,
                      spec_idx_std: float = .2


                      ) -> popsynth.PopulationSynth:

    r_max = 5.

    if use_clouds:

        # the host galaxy gas will be
        # created by embedding GRBs in
        # clouds

        host_nh = HostGas()
        host_nh.nh_mean = host_gas_mean
        host_nh.zratio = host_gas_cloud_ratio

    else:

        # the host gas will be drawn from
        # a log normal which is empirical

        host_nh = popsynth.aux_samplers.Log10NormalAuxSampler(
            name="host_nh", observed=False)
        host_nh.mu = host_gas_mean
        host_nh.sigma = 0.5

    # there are no random variables for the milky way gas
    mw_nh = MilkyWayGas()

    # GRB spectrum

    # sample the spectral index
    # of the power law

    spec_idx = popsynth.aux_samplers.NormalAuxSampler(
        name="spec_idx", observed=False)
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

    ls = ObscuredFluxSampler()

    ls.set_secondary_sampler(spec_idx)
    ls.set_secondary_sampler(powerlaw)
    ls.set_secondary_sampler(host_nh)
    ls.set_secondary_sampler(mw_nh)

    pop_gen: popsynth.PopulationSynth = popsynth.populations.SFRPopulation(
        r0=r0,
        rise=rise,
        decay=decay,
        peak=peak,
        r_max=r_max,



    )

    pop_gen.add_observed_quantity(ls)

    return pop_gen
