import collections
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import astropy.io.fits as fits
import h5py
import numpy as np
import threeML
from astromodels import Log_uniform_prior, Uniform_prior
from astromodels.utils.data_files import _get_data_file_path
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike
from tqdm.auto import tqdm

from whimstan.utils.colors import Colors

from ..utils import setup_logger
from ..utils.absori_precalc import AbsoriCalculations, sum_sigma_interp_precalc
from ..utils.format_conversions import (
    build_spectrum_like_from_hdf,
    plugin_to_hdf_group,
)
from .catalog import XRTCatalog, XRTObs

log = setup_logger(__name__)


threeML.silence_warnings()
threeML.update_logging_level("WARNING")


def build_tbabs_arg(ene) -> np.ndarray:

    file_name = _get_data_file_path(Path("xsect/xsect_tbabs_wilm.fits"))
    fxs = fits.open(file_name)
    dxs = fxs[1].data
    xsect_ene = dxs["ENERGY"]
    xsect_val = dxs["SIGMA"]

    return np.interp(ene, xsect_ene, xsect_val)


class Database:
    def __init__(
        self,
        grb_database: Dict[str, DispersionSpectrumLike],
        catalog: XRTCatalog,
        is_sim: bool = True,
    ):

        self._plugins: Dict[str, DispersionSpectrumLike] = grb_database

        self._catalog: XRTCatalog = catalog

        self._is_sim: bool = is_sim

    def create_sub_selection(self, selection: np.ndarray) -> "Database":
        """
        create a database from an array of bools
        that specifies the selection within the catalog
        """

        # first get the selection from the catalog

        new_cat: XRTCatalog = self._catalog.get_sub_selection(
            selection=selection
        )

        # now pick off the plugins from the new catalog
        # and pump them into the

        new_database = collections.OrderedDict()

        for k, v in new_cat.catalog.items():

            new_database[k] = self._plugins[k]

        return Database(new_database, new_cat, self._is_sim)

    @property
    def is_sim(self) -> bool:

        return self._is_sim

    @property
    def plugins(self) -> Dict[str, DispersionSpectrumLike]:

        return self._plugins

    @property
    def catalog(self) -> XRTCatalog:

        return self._catalog

    @classmethod
    def read(cls, file_name):
        """
        create a database from a file
        or HDF group

        :param cls:
        :type cls:
        :param file_name: name of the file
        :type file_name:
        :returns:

        """
        if isinstance(file_name, str):

            is_file = True

            f = h5py.File(file_name, "r")

        elif isinstance(file_name, h5py.Group):

            f = file_name

            is_file = False

        else:

            raise RuntimeError()

        catalog = XRTCatalog.from_file(f["catalog"])

        grb_database = OrderedDict()

        for grb in catalog.grbs:

            grb_database[grb] = build_spectrum_like_from_hdf(f[grb])

        if is_file:

            f.close()

        return cls(grb_database, catalog)

    def write(self, file_name):
        """
        write the database to a file or
        HDF group

        :param file_name:
        :type file_name:
        :returns:

        """

        if isinstance(file_name, str):

            is_file = True

            f = h5py.File(file_name, "w")

        elif isinstance(file_name, h5py.Group):

            f = file_name

            is_file = False

        else:

            raise RuntimeError()

        cat_grp = f.create_group("catalog")

        self._catalog.write(cat_grp)

        for grb in self.catalog.grbs:

            grb_grp = f.create_group(grb)

            plugin_to_hdf_group(self._plugins[grb], grb_grp)

        if is_file:

            f.close()

    @classmethod
    def from_fits_files(
        cls,
        file_name: str,
        catalog: XRTCatalog,
        cat_path="data",
        is_sim=True,
        clean: bool = False,
    ):

        grbs = catalog.grbs

        with h5py.File(file_name, "w") as f:

            for grb in tqdm(grbs, colour=Colors.green, desc="Reading GRBs"):

                cat_path = Path(cat_path)
                bpath = cat_path / f"grb{grb}"

                if not is_sim:

                    options = [f"{x}pc" for x in ["a", "b", "c"]]
                    options.extend([f"{x}wt" for x in ["a", "b", "c"]])

                    for opt in options:

                        try:

                            observation = bpath / f"{opt}.pi"
                            background = bpath / f"{opt}back.pi"
                            response = bpath / f"{opt}.rmf"
                            arf_file = bpath / f"{opt}.arf"

                            plugin = OGIPLike(
                                "tmp",
                                observation=observation,
                                background=background,
                                response=response,
                                arf_file=arf_file,
                            )

                            break

                        except ValueError:

                            pass
                    else:

                        raise RuntimeError(f"No data for GRB {grb}")

                else:
                    opt = "apc"

                    observation = bpath / f"{opt}.pha"
                    background = bpath / f"{opt}_bak.pha"
                    response = bpath / f"{opt}.rsp"
                    arf_file = None

                    plugin = OGIPLike(
                        "tmp",
                        observation=observation,
                        background=background,
                        response=response,
                        arf_file=arf_file,
                        spectrum_number=1,
                    )

                grb_grp = f.create_group(grb)

                plugin_to_hdf_group(plugin, grb_grp)

                if clean and is_sim:
                    observation.unlink()
                    background.unlink()
                    response.unlink()

                    bpath.rmdir()

            # Now save the catalog

            cat_grp = f.create_group("catalog")

            catalog.write(cat_grp)

        return cls.read(file_name=file_name)

    def build_stan_data(
        self,
        use_absori: bool = False,
        use_mw_gas: bool = True,
        use_host_gas: bool = True,
        k_offset: float = -10,
        nh_host_offset: float = 0.,
    ):

        """

        :param catalog:
        :type catalog:
        :param cat_path:
        :type cat_path:
        :param is_sim:
        :type is_sim:
        :param use_absori:
        :type use_absori:
        :param use_mw_gas:
        :type use_mw_gas:
        :param use_host_gas:
        :type use_host_gas:
        :returns:

        """

        z = []

        if use_mw_gas:
            nH_mw = []

        exposure_ratio = []
        counts = []
        bkg = []
        mask = []
        n_chans_used = []
        # rsp = []

        arf = []

        N_ene = []
        N_chan = []

        pca = []
        pcaz = []
        ene_avg = []
        ene_width = []
        exposure = []

        grbs = self._catalog.grbs

        N_grbs = len(grbs)

        for grb in tqdm(grbs, colour="#3DFF6C", desc="building GRBs"):

            z.append(self._catalog.catalog[grb].z)
            if use_mw_gas:
                nH_mw.append(self._catalog.catalog[grb].nH_mw)

            # extract data from the plugin associated with
            # this GRB plugin

            x: XRTObs = XRTObs.extract_xrt_data(self._plugins[grb])

            N_ene.append(int(x.n_ene))
            N_chan.append(int(x.n_chan))

            n_chans_used.append(int(x.n_chans_used))

            # the rmf does NOT change
            rmf = x.rsp.tolist()

            arf.append(x.arf.tolist())

            # rsp.append(x.rsp.tolist())
            exposure_ratio.append(float(x.scale_factor))
            counts.append(x.obs_count)
            bkg.append(x.bkg_count)

            mask.append(x.mask)

            exposure.append(x.exposure)
            ene_avg.append(x.ene_avg.tolist())
            ene_width.append(x.ene_width.tolist())

            if use_mw_gas:
                p = build_tbabs_arg(x.ene_avg).tolist()
                pca.append(p)

            if use_host_gas:
                pz = build_tbabs_arg(
                    x.ene_avg * (1 + self._catalog.catalog[grb].z)
                )
                pcaz.append(pz.tolist())

            # absori stuff

        # absori stuff
        if use_absori:

            # build the class that opens all the data

            absori_calc = AbsoriCalculations()

            sum_sigma_interp = np.zeros((N_grbs, N_ene[0], 10, 26))

            # calc ionizing spectrum - for fixed gamma=2 at the moment

            for i, zval in enumerate(z):
                sum_sigma_interp[i] = sum_sigma_interp_precalc(
                    zval,
                    np.array(ene_avg[i]),
                    absori_calc.energy,
                    absori_calc.sigma.T,
                    0.02,
                )

            absori_dict = dict(
                # absori
                spec=absori_calc.get_spec(),
                ion=absori_calc.ion,
                sigma=absori_calc.sigma,
                atomicnumber=absori_calc.atomic_number,
                sum_sigma_interp=sum_sigma_interp,
                abundance=absori_calc.get_abundance(),
                xi=1,  # fixed at the moment
            )

        res = dict(
            N_grbs=N_grbs,
            N_chan=N_chan[0],
            N_ene=N_ene[0],
            #            rsp=rsp,
            rmf=rmf,
            arf=arf,
            exposure_ratio=exposure_ratio,
            ene_avg=ene_avg,
            ene_width=ene_width,
            counts=counts,
            bkg=bkg,
            mask=mask,
            n_chans_used=n_chans_used,
            z=z,
            precomputed_absorp=pca,
            host_precomputed_absorp=pcaz,
            exposure=exposure,
            K_offset=k_offset,
            nh_host_offset=nh_host_offset
        )

        if use_mw_gas:
            res["nH_mw"] = nH_mw
            res["precomputed_absorp"] = pca

        if use_host_gas:
            res["host_precomputed_absorp"] = pcaz

        if use_absori:
            res.update(absori_dict)

        return res

    def build_3ml_analysis(
        self,
        id: int,
        with_whim: bool = False,
        integration_method: str = "trapz",
    ) -> threeML.BayesianAnalysis:

        grb: str = self._catalog.grbs[id]

        plugin: DispersionSpectrumLike = self._plugins[grb]

        plugin.set_active_measurements("0.3-10.")

        plugin.model_integrate_method = integration_method

        model_container = self._catalog.catalog[grb].get_spectrum(
            with_host=True, with_whim=with_whim
        )

        if with_whim:

            model = model_container.model_all
            model_name = "all"

            model.point_sources[
                model_name
            ].spectrum.main.composite.n0_4.prior = Log_uniform_prior(
                lower_bound=1e-9, upper_bound=1e-4
            )

            model.point_sources[
                model_name
            ].spectrum.main.composite.temp_4.prior = Log_uniform_prior(
                lower_bound=1e4, upper_bound=1e7
            )

        else:

            model = model_container.model_host
            model_name = "host"

        model.point_sources[
            model_name
        ].spectrum.main.composite.F_1.prior = Log_uniform_prior(
            lower_bound=1e-20, upper_bound=1e-2
        )
        model.point_sources[
            model_name
        ].spectrum.main.composite.index_1.prior = Uniform_prior(
            lower_bound=-3, upper_bound=-1
        )
        model.point_sources[
            model_name
        ].spectrum.main.composite.NH_3.prior = Log_uniform_prior(
            lower_bound=1.0e19 / 1.0e22, upper_bound=1.0e26 / 1.0e22
        )

        return threeML.BayesianAnalysis(model, threeML.DataList(plugin))
