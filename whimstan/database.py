from dataclasses import dataclass
from typing import Dict

from pathlib import Path
from collections import OrderedDict

import numpy as np

import h5py

from tqdm.auto import tqdm

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike
from threeML import silence_warnings, update_logging_level
from astromodels.utils.data_files import _get_data_file_path


from .coverters import plugin_to_hdf_group, build_spectrum_like_from_hdf
from .catalog import XRTCatalog

from .absori_precalc import (
    get_abundance,
    load_absori_base,
    get_spec,
    sum_sigma_interp_precalc,
)
from whimstan import catalog


silence_warnings()
update_logging_level("WARNING")


def build_tbabs_arg(ene):

    file_name = _get_data_file_path(Path("xsect/xsect_tbabs_wilm.fits"))
    fxs = fits.open(file_name)
    dxs = fxs[1].data
    xsect_ene = dxs["ENERGY"]
    xsect_val = dxs["SIGMA"]

    return np.interp(ene, xsect_ene, xsect_val)


@dataclass
class XRTObs:

    n_ene: int
    n_chan: int
    ene_avg: np.array
    ene_width: np.array
    obs_count: List[int]
    bkg_count: List[int]
    mask: np.array
    n_chans_used: int
    scale_factor: float
    rsp: np.array
    exposure: float


def extract_xrt_data(plugin):

    plugin.set_active_measurements("0.3-10.")

    n_chans_used = sum(plugin.mask)

    mask = np.zeros(len(plugin.mask))

    mask[:n_chans_used] = np.where(plugin.mask)[0] + 1  # plus one for Stan

    n_ene = len(plugin.response.monte_carlo_energies) - 1
    n_chan = len(plugin.response.ebounds) - 1
    rsp = plugin.response.matrix
    scale_factor = plugin.scale_factor

    ene_lo = plugin.response.monte_carlo_energies[:-1]
    ene_hi = plugin.response.monte_carlo_energies[1:]

    ene_avg = (ene_hi + ene_lo) / 2.0
    ene_width = ene_hi - ene_lo

    return XRTObs(
        n_ene=n_ene,
        n_chan=n_chan,
        ene_avg=ene_avg,
        ene_width=ene_width,
        obs_count=[int(x) for x in plugin.observed_counts],
        bkg_count=[int(x) for x in plugin.background_counts],
        rsp=rsp,
        scale_factor=float(scale_factor),
        mask=[int(x) for x in mask],
        n_chans_used=int(n_chans_used),
        exposure=float(plugin.exposure),
    )


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
        :param file_name:
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

            f = h5py.File(file_name, "r")

        elif isinstance(file_name, h5py.Group):

            f = file_name

            is_file = False

        else:

            raise RuntimeError()

        cat_grp = f.create_group("catalog")

        self._catalog.to_file(cat_grp)

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

            for grb in tqdm(grbs, colour="#3DFF6C", desc="Reading GRBs"):

                cat_path = Path(cat_path)
                bpath = cat_path / f"grb{grb}"

                if not is_sim:

                    options = [f"{x}pc" for x in ["a", "b", "c"]]
                    options.extend([f"{x}wt" for x in ["a", "b", "c"]])

                    for opt in options:

                        try:

                            observation = (bpath / f"{opt}.pi",)
                            background = (bpath / f"{opt}back.pi",)
                            response = (bpath / f"{opt}.rmf",)
                            arf_file = (bpath / f"{opt}.arf",)

                            print(f"GRB {grb} using {opt}")

                            break

                        except:

                            pass
                    else:

                        raise RuntimeError(f"No data for GRB {grb}")

                    plugin = OGIPLike(
                        "tmp",
                        observation=observation,
                        background=background,
                        response=response,
                        arf_file=arf_file,
                    )

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

            # Now save the catalog

            cat_grp = f.create_group("catalog")

            catalog.to_file(cat_grp)

        return cls.read(file_name=file_name)

    def build_stan_data(
        self,
        use_absori=False,
        use_mw_gas=True,
        use_host_gas=True,
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
        N_grbs = len(grbs)
        z = []
        if use_mw_gas:
            nH_mw = []
        exposure_ratio = []
        counts = []
        bkg = []
        mask = []
        n_chans_used = []
        rsp = []

        N_ene = []
        N_chan = []

        pca = []
        pcaz = []
        ene_avg = []
        ene_width = []
        exposure = []

        grbs = self._catalog.grbs

        for grb in tqdm(grbs, colour="#3DFF6C", desc="building GRBs"):

            z.append(self._catalog.catalog[grb].z)
            if use_mw_gas:
                nH_mw.append(self._catalog.catalog[grb].nH_mw)

            # extract data from the plugin associated with
            # this GRB plugin

            x = extract_xrt_data(self._plugins[grb])

            N_ene.append(int(x.n_ene))
            N_chan.append(int(x.n_chan))

            n_chans_used.append(int(x.n_chans_used))

            rsp.append(x.rsp.tolist())
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

            ion, sigma, atomicnumber, absori_base_energy = load_absori_base()
            abundance = get_abundance()
            sum_sigma_interp = np.zeros((N_grbs, N_ene[0], 10, 26))

            # calc ionizing spectrum - for fixed gamma=2 at the moment
            spec = get_spec()

            for i, zval in enumerate(z):
                sum_sigma_interp[i] = sum_sigma_interp_precalc(
                    zval,
                    np.array(ene_avg[i]),
                    absori_base_energy,
                    sigma.T,
                    0.02,
                )

            absori_dict = dict(
                # absori
                spec=spec,
                ion=ion,
                sigma=sigma,
                atomicnumber=atomicnumber,
                sum_sigma_interp=sum_sigma_interp,
                abundance=abundance,
                xi=1,  # fixed at the moment
            )

        res = dict(
            N_grbs=N_grbs,
            N_chan=N_chan[0],
            N_ene=N_ene[0],
            rsp=rsp,
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
        )

        if use_mw_gas:
            res["nH_mw"] = nH_mw
            res["precomputed_absorp"] = pca

        if use_host_gas:
            res["host_precomputed_absorp"] = pcaz

        if use_absori:
            res.update(absori_dict)

        return res
