from typing import Dict

from pathlib import Path
from collections import OrderedDict
from threeML.plugins.OGIPLike import OGIPLike

import h5py

from tqdm.auto import tqdm

from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.OGIPLike import OGIPLike


from .coverters import plugin_to_hdf_group, build_spectrum_like_from_hdf
from .catalog import XRTCatalog


class Database:
    def __init__(
        self,
        grb_database: Dict[str, DispersionSpectrumLike],
        catalog: XRTCatalog,
    ):

        self._plugins: Dict[str, DispersionSpectrumLike] = grb_database

        self._catalog: XRTCatalog = catalog

    @property
    def plugins(self) -> Dict[str, DispersionSpectrumLike]:

        return self._plugins

    @property
    def catalog(self) -> XRTCatalog:

        return self._catalog

    @classmethod
    def read(cls, file_name):

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

        return cls(grb_database, catalog)

    def write(self):

        pass

    @classmethod
    def from_fits_files(
        cls,
        file_name: str,
        catalog: XRTCatalog,
        cat_path="data",
        is_sim=True,
        clean: bool = True,
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

            # Now save the catalog

            cat_grp = f.create_group("catalog")

            catalog.to_file(cat_grp)

        return cls.read(file_name=file_name)
