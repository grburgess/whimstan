import h5py

from threeML.utils.spectrum.binned_spectrum import (
    BinnedSpectrum,
    BinnedSpectrumWithDispersion,
)


from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.utils.OGIP.response import InstrumentResponse


def plugin_to_hdf_group(plugin: DispersionSpectrumLike, hdf_group: h5py.Group):
    """ """

    with plugin._without_mask_nor_rebinner():

        hdf_group.attrs["exposure"] = plugin.exposure
        hdf_group.attrs["bak_scale"] = plugin.background_scale_factor
        hdf_group.attrs["obs_scale"] = plugin.observed_spectrum.scale_factor

        hdf_group.create_dataset(
            "counts", data=plugin.observed_counts, compression="gzip"
        )

        hdf_group.create_dataset(
            "bkg_counts", data=plugin.background_counts, compression="gzip"
        )

    rsp_grp = hdf_group.create_group("response")

    rsp_grp.create_dataset(
        "matrix", data=plugin.response.matrix, compression="gzip"
    )

    rsp_grp.create_dataset(
        "ebounds", data=plugin.response.ebounds, compression="gzip"
    )

    rsp_grp.create_dataset(
        "mc_energies",
        data=plugin.response.monte_carlo_energies,
        compression="gzip",
    )


def binned_spectrum_from_hdf(hdf: h5py.Group, ebounds) -> BinnedSpectrum:

    return BinnedSpectrum(
        counts=hdf["bkg_counts"][()],
        exposure=hdf.attrs["exposure"][()],
        is_poisson=True,
        ebounds=ebounds,
        quality=None,
        scale_factor=hdf.attrs["bak_scale"],
        mission="Swift",
        instrument="XRT",
    )


def binned_dispersion_spectrum_from_hdf(
    hdf: h5py.Group, response
) -> BinnedSpectrumWithDispersion:

    return BinnedSpectrumWithDispersion(
        counts=hdf["counts"][()],
        exposure=hdf.attrs["exposure"],
        response=response,
        scale_factor=hdf.attrs["obs_scale"],
        is_poisson=True,
        quality=None,
        mission="Swift",
        instrument="XRT",
    )


def build_spectrum_like_from_hdf(hdf_grp: h5py.Group) -> DispersionSpectrumLike:

    # first get the response

    response_grp: h5py.Group = hdf_grp["response"]

    response = InstrumentResponse(
        matrix=response_grp["matrix"][()],
        ebounds=response_grp["ebounds"][()],
        monte_carlo_energies=response_grp["mc_energies"][()],
    )

    # get the source group

    spectrum = binned_dispersion_spectrum_from_hdf(hdf_grp, response=response)

    # get the background

    background = binned_spectrum_from_hdf(hdf_grp, ebounds=response.ebounds)

    return DispersionSpectrumLike(
        "grb", observation=spectrum, background=background
    )
