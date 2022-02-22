import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import arviz as av
from omegaconf import MISSING, OmegaConf

from .database import Database
from .fit import Fit
from .stan_code.stan_models import StanModel, get_model
from .utils import setup_logger

log = setup_logger(__name__)


@dataclass
class ModelParameters:
    name: str = MISSING
    t_whim_lower: float = 4.5
    t_whim_upper: float = 8
    t_whim_mu: float = 6.0
    t_whim_sigma: float = 1.0

    k_offset: float = -10.0
    nh_host_offset: float = 0.0

    host_alpha_mu: float = -1.0
    host_alpha_sigma: float = 0.5


@dataclass
class ModelSetup:
    use_host_gas: bool = True
    use_mw_gas: bool = True
    use_absori: bool = False
    model: ModelParameters = ModelParameters()


@dataclass
class FitParams:
    seed: int = 1234
    inits: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "log_nH_host_mu_raw": 0,
            "log_nH_host_sigma": 0.5,
            "log_K_mu_raw": -1,
            "index_mu": -2.0,
            "host_alpha": -1.0,
        }
    )
    iter_warmup: int = 1000
    iter_sampling: int = 500
    max_treedepth: int = 12


@dataclass
class FitSetup:
    use_advi: bool = False
    n_chains: int = 2
    n_threads: Optional[int] = None
    fit_params: FitParams = FitParams()


@dataclass
class FitInput:
    database: str = MISSING
    file_name: str = MISSING
    fit_setup: FitSetup = FitSetup()
    model_setup: ModelSetup = ModelSetup()


class Fitter:
    def __init__(self, input: FitInput) -> None:

        self._config: FitInput = input

    @property
    def config(self) -> FitInput:
        return self._config

    @classmethod
    def from_file(cls, file_name: str) -> "Fitter":

        base_input = OmegaConf.structured(FitInput)

        file_input = OmegaConf.load(file_name)

        return cls(OmegaConf.merge(base_input, file_input))

    def make_fit(
        self,
        save_stan_fit: bool = True,
        clean_model: bool = False,
        use_opencl: bool = False,
        opt_level: Union[int, str] = 0,
    ):

        """ """

        database = Database.read(self._config.database)

        if self._config.fit_setup.n_threads is None:

            n_threads = len(database.plugins)

        else:

            n_threads = self._config.fit_setup.n_threads

        n_chains = self._config.fit_setup.n_chains

        model: StanModel = get_model(self._config.model_setup.model.name)

        cur_dir = Path().cwd()

        model.build_model(use_opencl=use_opencl, opt_level=opt_level)

        if clean_model:

            model.clean_model()
            model.build_model(use_opencl=use_opencl, opt_level=opt_level)

        log.info("building data")
        data = database.build_stan_data(
            use_absori=self._config.model_setup.use_absori,
            use_mw_gas=self._config.model_setup.use_mw_gas,
            use_host_gas=self._config.model_setup.use_host_gas,
            k_offset=self._config.model_setup.model.k_offset,
            nh_host_offset=self._config.model_setup.model.nh_host_offset,
            t_whim_lower=self._config.model_setup.model.t_whim_lower,
            t_whim_upper=self._config.model_setup.model.t_whim_upper,
            t_whim_mu=self._config.model_setup.model.t_whim_mu,
            t_whim_sigma=self._config.model_setup.model.t_whim_sigma,
        )


        if self._config.fit_setup.use_advi:

            log.info("launching ADVI initialization")

            vb_fit = model.model.variational(data=data,
                              require_converged=False, seed=123)

            for k,v in vb_fit.stan_variables():

                log.info(f"{k}: {v}")

            self._config.fit_setup.fit_params.inits = vb_fit.stan_variables()
        

        log.info("launching fit")

        stan_fit = model.model.sample(
            data=data,
            chains=n_chains,
            parallel_chains=n_chains,
            threads_per_chain=n_threads,
            show_progress=True,
            **self._config.fit_setup.fit_params,
        )

        os.chdir(cur_dir)

        # transfer fit to arviz
        log.info("converting fit to arviz")
        av_fit = av.from_cmdstanpy(stan_fit)

        file_name = self._config.file_name

        if save_stan_fit:

            av_fit.to_netcdf(f"stan_fit_{file_name}")

            log.info(f"saved Stan fit to stan_fit_{file_name}")

        fit = Fit.from_live_fit(
            av_fit,
            database=database,
            model_name=self._config.model_setup.model.name,
        )

        fit.write(file_name=file_name)

        log.info(f"saved fit to {file_name}")
