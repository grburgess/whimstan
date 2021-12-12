from IPython.display import YouTubeVideo, display

from ._version import get_versions
#from .prep import build_stan_data, build_tbabs_arg

from ,fitter import make_fit
from .fit import Fit

from .database import Database

from .stan_code.stan_models import get_model
from .catalog import XRTCatalog, XRTCatalogEntry


# display(YouTubeVideo("Tb6tz6ohprw", start=17, autoplay=1))


__version__ = get_versions()['version']
del get_versions
