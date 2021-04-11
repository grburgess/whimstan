from pathlib import Path
from typing import Any, Dict, List, Optional

import arviz as av
import numpy as np
from natsort import natsorted

from .catalog import XRTCatalog


class Fit(object):

    def __init__(self, catalog: XRTCatalog, stan_fit: av.data.InferenceData, data_path: Optional[Path] = None):
        """
        :param catalog:
        ::

        """
        self._catalog: XRTCatalog = catalog
        self._data_path: Path = data_path

        self._n_grbs: int = stan_fit.posterior.K.stack(
            sample=("chain", "draw")).values.shape[0]

        self._flux: np.ndarray = stan_fit.posterior.K.stack(
            sample=("chain", "draw")).values
        self._index: np.ndarray = stan_fit.posterior.index.stack(
            sample=("chain", "draw")).values
        self._host_nh: np.ndarray = stan_fit.posterior.nH_host.stack(
            sample=("chain", "draw")).values

        # whim stuff

        # group properties

        self._log_nh_host_mu: np.ndarray = stan_fit.posterior.log_nH_host_mu.stack(
            sample=("chain", "draw")).values

        self._log_nh_host_sigma: np.ndarray = stan_fit.posterior.log_nH_host_sigma.stack(
            sample=("chain", "draw")).values

        self._index_mu: np.ndarray = stan_fit.posterior.index_mu.stack(
            sample=("chain", "draw")).values

        self._index_sigma: np.ndarray = stan_fit.posterior.index_sigma.stack(
            sample=("chain", "draw")).values

        if data_path is not None:

            self._grbs = [x.name.replace("grb", "")
                          for x in natsorted(data_path.glob("grb*"))]

    @property
    def catalog(self) -> XRTCatalog:
        self._catalog

    @property
    def n_grbs(self) -> int:
        return self._n_grbs

    @property
    def z(self) -> np.ndarray:
        return self._catalog.z

    @property
    def flux(self) -> np.ndarray:
        return self._flux

    @property
    def index(self) -> np.ndarray:
        return self._index

    @property
    def host_nh(self) -> np.ndarray:
        return self._host_nh

    @property
    def index_mu(self) -> np.ndarray:
        return self._index_mu

    @property
    def index_sigma(self) -> np.ndarray:
        return self._index_sigma

    @property
    def log_nh_host_mu(self) -> np.ndarray:
        return self._log_nh_host_mu

    @property
    def log_nh_host_sigma(self) -> np.ndarray:
        return self._log_nh_host_sigma
