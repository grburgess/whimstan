from pathlib import Path

from whimstan import Database, Fit, SpectrumFactory, make_fit


def test_fit(tiny_population):

    factory = SpectrumFactory(tiny_population)

    factory.create_database("tiny_db.h5")

    db = Database.read("tiny_db.h5")

    fit_params = dict(
        seed=3214,
        inits={
            "log_nH_host_mu_raw": 0,
            "log_nH_host_sigma": 0.5,
            "log_K_mu_raw": -1,
            "index_mu": -2.0,
            "host_alpha": -2.0,
        },
        iter_warmup=1,
        iter_sampling=5,
        max_treedepth=9,
    )

    make_fit(
        "no_whim",
        db,
        n_threads=1,
        fit_params=fit_params,
        file_name="fit_no_whim.h5",
        n_chains=1,
        # use_opencl=True,
        use_host_gas=True,
        use_mw_gas=True,
        use_absori=False,
        clean_model=True,
    )

    fit: Fit = Fit.from_file("fit_no_whim.h5")

    fit.plot_data_spectrum(1)

    fit.plot_flux_distribution()

    fit.plot_model_spectrum(1)

    fit.plot_ppc(1, n_sims=1)
