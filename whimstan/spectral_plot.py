import matplotlib.pyplot as plt
import numpy as np
import threeML

# from threeML import threeML_config
# from threeML.io.plotting.cmap_cycle import cmap_intervals
from threeML.io.plotting.data_residual_plot import ResidualPlot

# from threeML_utils.colors import Colors

NO_REBIN = 1e-99


# def scale_colour(self, colour, scalefactor):    # pragma: no cover
#     if isinstance(colour, np.ndarray):
#         r, g, b = colour[:3] * 255.0
#     else:
#         hexx = colour.strip('#')
#         if scalefactor < 0 or len(hexx) != 6:
#             return hexx
#         r, g, b = int(hexx[:2], 16), int(hexx[2:4], 16), int(hexx[4:], 16)
#     r = self._clamp(int(r * scalefactor))
#     g = self._clamp(int(g * scalefactor))
#     b = self._clamp(int(b * scalefactor))
#     return "#%02x%02x%02x" % (r, g, b)


def display_posterior_model_counts(
    plugin,
    model,
    samples,
    data_color="k",
    model_color="r",
    thin=100,
    min_rate=1,
    shade=True,
    q_level=68,
    gradient=0.6,
    axes=None,
    **kwargs
):

    show_residuals = False

    if axes != None:
        residual_plot = ResidualPlot(
            show_residuals=show_residuals, model_subplot=axes
        )
    else:
        residual_plot = ResidualPlot(show_residuals=show_residuals)
        axes = residual_plot.data_axis

    plugin.set_model(model)

    show_legend = False

    for params in samples:

        model.set_free_parameters(params)

        # for i, (k, v) in enumerate(model.free_parameters.items()):

        #     v.value = params[i]

        # first with no data
        if shade:
            per_det_y = []
            per_det_x = []

        if not shade:

            plugin.display_model(
                data_color=data_color,
                model_color=model_color,
                min_rate=min_rate,
                step=False,
                show_residuals=False,
                show_data=False,
                show_legend=show_legend,
                ratio_residuals=False,
                model_label=None,
                model_subplot=axes,
                model_kwargs=dict(alpha=0.1),
                **kwargs
                #                model_subplot=axes,
                #                data_kwargs=data_kwargs,
            )
    #     else:

    #         # this is private for now
    #         rebinned_quantities = plugin._construct_counts_arrays(min_rate, ratio_residuals)

    #         if step:

    #             pass

    #         else:

    #             y = (rebinned_quantities['expected_model_rate'] / rebinned_quantities['chan_width'])[plugin.mask]

    #             x = np.mean(
    #                 [rebinned_quantities['energy_min'], rebinned_quantities['energy_max']], axis=0)[plugin.mask]

    #             per_det_y.append(y)
    #             per_det_x.append(x)

    # if shade:
    #     shade_y.append(per_det_y)
    #     shade_x.append(per_det_x)

    # if shade:

    #     # convert to per detector
    #     shade_y = np.array(shade_y).T
    #     shade_x = np.array(shade_x).T
    #     model_kwargs.pop('zorder')
    #     for key, data_color, model_color, min_rate, model_label, x, y, in zip(
    #             data_keys, data_colors, model_colors, min_rates, model_labels, shade_x, shade_y):

    #         # we have to do a little reshaping because... life

    #         y = np.array([yy.tolist() for yy in y])

    #         q_levels = np.atleast_1d(q_level)
    #         q_levels.sort()

    #         scale = 1.
    #         zorder = -100
    #         for level in q_levels:

    #             color = color_config.format(model_color)
    #             color_scale = color_config.scale_colour(color, scale)

    #             # first we need to get the quantiles along the energy axis
    #             low = np.percentile(y, 50 - level * 0.5, axis=0)
    #             high = np.percentile(y, 50 + level * 0.5, axis=0)

    #             residual_plot.data_axis.fill_between(x[0], low, high, color=color_scale, zorder=zorder, **model_kwargs)

    #             scale *= gradient

    #             zorder -= 1

    # for key, data_color, model_color, min_rate, model_label in zip(data_keys, data_colors, model_colors, min_rates,
    #                                                                model_labels):

    #     # NOTE: we use the original (unmasked) vectors because we need to rebin ourselves the data later on

    #     data = bayesian_analysis.data_list[key]    # type: threeML.plugins.SpectrumLike.SpectrumLike

    plugin.display_model(
        data_color=data_color,
        model_color=model_color,
        min_rate=min_rate,
        step=True,
        show_residuals=False,
        show_data=True,
        show_legend=show_legend,
        ratio_residuals=False,
        #        model_label=model_label,
        model_subplot=axes,
        model_kwargs=dict(alpha=0.0),  # no model
        **kwargs
        #    data_kwargs=data_kwargs
    )

    return residual_plot.figure
