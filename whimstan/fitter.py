from typing import Dict, Any, Optional

import arviz as av

from .database import Database
from .fit import Fit

from .stan_code.stan_models import get_model, StanModel
from whimstan import database


def make_fit(
    model_name: str,
    database: Database,
    fit_params: Dict[str, Any],
    file_name: str,
    n_threads: Optional[int] = None,
    n_chains: int = 2,
    use_absori: bool = False,
    use_mw_gas: bool = True,
    use_host_gas: bool = True,
    save_stan_fit: bool = True,
    clean_model: bool = False
):

    if n_threads is None:

        n_threads = len(database.plugins)

    model: StanModel = get_model(model_name)

    if clean_model:

        model.clean_model()


    model.build_model()


    data = database.build_stan_data(
        use_absori=use_absori, use_mw_gas=use_mw_gas, use_host_gas=use_host_gas
    )

    stan_fit = model.model.sample(
        data=data,
        chains=n_chains,
        parallel_chains=n_chains,
        threads_per_chain=n_threads,
        show_progress=True,
        **fit_params,
    )

    # transfer fit to arviz

    av_fit = av.from_cmdstanpy(stan_fit)

    if save_stan_fit:

        av_fit.to_netcdf(f"stan_fit_{file_name}")

    fit = Fit.from_live_fit(av_fit, database=database, model_name=model_name)

    fit.write(file_name=file_name)
