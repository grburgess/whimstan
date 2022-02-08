from pathlib import Path

from whimstan import Database, Fit, Fitter, SpectrumFactory
from whimstan.utils import get_path_of_data_file


def test_fit(tiny_population):

    factory = SpectrumFactory(tiny_population)

    factory.create_database("tiny_db.h5")

    fitter = Fitter.from_file(get_path_of_data_file("no_whim_demo.yml"))

    fitter.config.database = "tiny_db.h5"
    fitter.config.fit_setup.fit_params.iter_warmup = 1
    fitter.config.fit_setup.fit_params.iter_sampling = 5
    fitter.config.fit_setup.fit_params.max_treedepth = 9
    fitter.config.fit_setup.n_chains = 1
    fitter.config.fit_setup.n_threads = 1

    fitter.config.file_name = "fit_no_whim.h5"

    fitter.make_fit(
        clean_model=True,
    )

    fit: Fit = Fit.from_file("fit_no_whim.h5")

    fit.plot_data_spectrum(1)

    fit.plot_flux_distribution()

    fit.plot_model_spectrum(1)

    fit.plot_ppc(1, n_sims=1)
