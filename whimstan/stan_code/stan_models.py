import cmdstanpy
import pkg_resources
import os
_available_models = {}

_available_models["simple_xrt"] = "simple_xrt.stan"

# HBM with galaxy gas and plaw
_available_models["hbm_xrt"] = "host_hbm.stan"

# HBM with ONLY plaw`
_available_models["hbm_plaw_xrt"] = "hbm_plaw_only.stan"


# t_fixed => T=10**7 K at the moment!
_available_models["whim_only_t_fixed"] = "whimonly_t_fixed.stan"
_available_models["whim_and_mw_t_fixed"] = "whim_and_mw_t_fixed.stan"
_available_models["all_t_fixed"] = "all_t_fixed.stan"

# This is the model with mw, host and whim component and the n0 and the temp
# is free for whim
_available_models["all"] = "all.stan"

class StanModel(object):
    def __init__(self, name, stan_file):

        self._name = name
        self._stan_file = pkg_resources.resource_filename(
            "whimstan", os.path.join("stan_code", stan_file)
        )

        self._model = None

    def build_model(self):
        """
        build the stan model

        :returns:
        :rtype:

        """

        cpp_options = dict(STAN_THREADS=True)

        self._model = cmdstanpy.CmdStanModel(
            stan_file=self._stan_file, model_name=self._name, cpp_options=cpp_options
        )

    @property
    def model(self):
        return self._model

    def clean_model(self):
        """
        Clean the model bin file
        to allow for compiling

        :returns:
        :rtype:

        """

        if self._model is not None:

            os.remove(self._model.exe_file)


def get_model(model_name):
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
