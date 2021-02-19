from .prep import build_stan_data, XRTCatalog

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
