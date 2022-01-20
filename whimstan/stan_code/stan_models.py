import cmdstanpy
import pkg_resources
import os
from pathlib import Path

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

# This is the model with mw, host and whim component and the n0 and the temp
# is free for whim
_available_models["all"] = "all.stan"
_available_models["whim_skew"] = "whim_skew.stan"


_available_models["whim"] = "whim.stan"
_available_models["no_whim"] = "no_whim.stan"


class StanModel:
    def __init__(self, name: str, stan_file: str):

        self._name = name
        self._stan_file = pkg_resources.resource_filename(
            "whimstan", os.path.join("stan_code", stan_file)
        )

        # self._hpp_file = pkg_resources.resource_filename(
        #     "whimstan", os.path.join("stan_code", file_stem, ".hpp")
        # )

        # self._o_file = pkg_resources.resource_filename(
        #     "whimstan", os.path.join("stan_code", file_stem, ".o")
        # )


        self._model = None

    def build_model(self, use_opencl=False, opt=True):
        """
        build the stan model

        :returns:
        :rtype:

        """

        cpp_options = dict(STAN_THREADS=True)


        if use_opencl:

            stanc_options = dict('use_opencl'=True)

            cpp_options["STAN_OPENCL"] = True
            cpp_options["OPENCL_DEVICE_ID"] = 0
            cpp_options["OPENCL_PLATFORM_ID"] = 0

        if opt:

            cpp_options["STAN_CPP_OPTIMS"] = True
            cpp_options["STAN_NO_RANGE_CHECKS"] = True



        # get the current working dir

        cur_dir = Path.cwd()

        self._model = cmdstanpy.CmdStanModel(
            stan_file=self._stan_file,
            model_name=self._name,
            cpp_options=cpp_options,
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

            Path(self._model.exe_file).unlink()

            # try:

            #     Path(self._hpp_file).unlink()

            # except:

            #     pass

            # try:

            #     Path(self._o_file).unlink()

            # except:

            #     pass


def get_model(model_name) -> StanModel:
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
