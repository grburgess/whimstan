from threeML.io.plotting.data_residual_plot import ResidualPlot

NO_REBIN = 1e-99


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

    if axes is not None:
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
