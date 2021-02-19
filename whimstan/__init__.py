from ._version import get_versions
from .prep import XRTCatalog, build_stan_data
from .stan_code.stan_models import get_model

__version__ = get_versions()['version']
del get_versions
