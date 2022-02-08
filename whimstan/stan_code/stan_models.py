import os
from pathlib import Path
from typing import Optional, Union

import cmdstanpy
import pkg_resources

from ..utils import setup_logger

log = setup_logger(__name__)


_available_models = {}

_available_models["simple_xrt"] = "simple_xrt.stan"

# HBM with galaxy gas and plaw
_available_models["hbm_xrt"] = "host_hbm.stan"

_available_models["hbm_xrt_skew"] = "host_hbm_skew.stan"


# HBM with ONLY plaw`
_available_models["hbm_plaw_xrt"] = "hbm_plaw_only.stan"


# t_fixed => T=10**7 K at the moment!
_available_models["whim_only_t_fixed"] = "whimonly_t_fixed.stan"
_available_models["whim_and_mw_t_fixed"] = "whim_and_mw_t_fixed.stan"
_available_models["all_t_fixed"] = "all_t_fixed.stan"


_available_models["whim"] = "whim.stan"
_available_models["no_whim"] = "no_whim.stan"
_available_models["no_whim_limit"] = "no_whim_limit.stan"


class StanModel:
    def __init__(self, name: str, stan_file: str):

        """

        :param name:
        :type name: str
        :param stan_file:
        :type stan_file: str
        :returns:

        """
        self._name = name
        self._stan_file = pkg_resources.resource_filename(
            "whimstan", os.path.join("stan_code", stan_file)
        )

        file_stem = Path(self._stan_file).stem

        self._hpp_file = pkg_resources.resource_filename(
            "whimstan", os.path.join("stan_code", file_stem, ".hpp")
        )

        self._o_file = pkg_resources.resource_filename(
            "whimstan", os.path.join("stan_code", file_stem, ".o")
        )
        log.info(f"creating stan model {name} from {stan_file}")

        self._model = None

    def build_model(
        self,
        use_opencl: bool = False,
        opt: bool = True,
        opt_level: Optional[Union[str, int]] = None,
    ):
        """
        build the stan model

        :returns:
        :rtype:

        """

        cpp_options = dict(STAN_THREADS=True)

        stanc_options = {}

        if use_opencl:

            stanc_options['use-opencl'] = True

            cpp_options["STAN_OPENCL"] = True
            cpp_options["OPENCL_DEVICE_ID"] = 0
            cpp_options["OPENCL_PLATFORM_ID"] = 0

        if opt:

            cpp_options["STAN_CPP_OPTIMS"] = True
            cpp_options["STAN_NO_RANGE_CHECKS"] = True

        # it is is zero, then we want to be safe for
        # older cmdstan
        if (opt_level is not None) and (opt_level != 0):

            stanc_options[f"O{opt_level}"] = True

        # get the current working dir

        cur_dir = Path.cwd()

        log.info(f"compiling {self._name}")
        log.info(f"cpp: {cpp_options}")
        log.info(f"stanc: {stanc_options}")

        self._model = cmdstanpy.CmdStanModel(
            stan_file=self._stan_file,
            model_name=self._name,
            cpp_options=cpp_options,
            stanc_options=stanc_options,
        )

        os.chdir(cur_dir)

    @property
    def model(self) -> cmdstanpy.CmdStanModel:
        return self._model

    def clean_model(self) -> None:
        """
        Clean the model bin file
        to allow for compiling

        :returns:
        :rtype:

        """

        if self._model is not None:

            log.info(f"removing: {self._model.exe_file}")

            Path(self._model.exe_file).unlink()

            if Path(self._hpp_file).exists():

                log.info(f"removing: {self._hpp_file}")

                Path(self._hpp_file).unlink()

            if Path(self._o_file).exists():

                log.info(f"removing: {self._o_file}")

                Path(self._o_file).unlink()


def get_model(model_name: str) -> StanModel:
    """
    Retrieve the stan model

    :param model_name:
    :returns:
    :rtype:

    """

    assert (
        model_name in _available_models
    ), f"please chose {','.join(x for x in _available_models.keys()) }"

    return StanModel(model_name, _available_models[model_name])
