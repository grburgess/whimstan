import popsynth
import numba as nb
import numpy as np
from gdpyc import GasMap, DustMap
from pathlib import Path


from whimstan.population_generator import create_simulation

from whimstan.utils.package_data import get_path_of_data_file


from whimstan import Database, XRTCatalog


from threeML import OGIPLike

import astropy.units as u

import pytest

np.seterr(all="ignore")
import warnings

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def demo_plugin():



    demo_plugin = OGIPLike(
        "tmp",
        observation=get_path_of_data_file("apcsource.pi"),
        background=get_path_of_data_file("apcback.pi"),
    response=get_path_of_data_file("apc.rmf"),
    arf_file=get_path_of_data_file("apc.arf"),
    verbose=False,
    )


    yield demo_plugin



@pytest.fixture(scope="session")
def no_whim_population(demo_plugin):






    pop_gen = create_simulation(
        r0=2.5,
        z_max=10,
        Lmin=1e48,
        alpha=1.5,
        host_gas_mean=22,
        host_gas_sigma=0.5,
        host_gas_cloud_ratio=0.1,
        use_clouds=True,
        vari_clouds=True,
        b_limit=10.0,
        mw_nh_limit=21.0,
        demo_plugin=demo_plugin,
        counts_limit=500,
        exposure_low=900,
        exposure_high=1000,
    )

    # flux_selector = popsynth.HardFluxSelection()

    # flux_selector.boundary = 1e-11


    # pop_gen.set_flux_selection(flux_selector)

    pop = pop_gen.draw_survey()

    pop = pop.to_sub_population()

    yield pop


@pytest.fixture(scope="session")
def whim_population(demo_plugin):






    pop_gen = create_simulation(
        r0=2.5,
        z_max=10,
        Lmin=1e48,
        alpha=1.5,
        host_gas_mean=22,
        host_gas_sigma=0.5,
        host_gas_cloud_ratio=0.1,
        use_clouds=True,
        vari_clouds=True,
        b_limit=10.0,
        mw_nh_limit=21.0,
        whim_n0=1e-7,
        whim_T=1e7,
        demo_plugin=demo_plugin,
        counts_limit=500,
        exposure_low=900,
        exposure_high=1000,
    )

    # flux_selector = popsynth.HardFluxSelection()

    # flux_selector.boundary = 1e-11


    # pop_gen.set_flux_selection(flux_selector)

    pop = pop_gen.draw_survey()

    pop = pop.to_sub_population()

    yield pop

