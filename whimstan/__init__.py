from ._version import get_versions
from .database import Database, XRTCatalog, XRTCatalogEntry
from .fit import Fit
from .fitter import make_fit
from .simulations import SpectrumFactory, create_population
from .stan_code.stan_models import get_model
# display(YouTubeVideo("Tb6tz6ohprw", start=17, autoplay=1))


__version__ = get_versions()["version"]
del get_versions
