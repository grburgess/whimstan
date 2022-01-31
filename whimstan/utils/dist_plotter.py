import matplotlib.pyplot as plt
import numpy as np


def dist_plotter(x, y, ax, ci=0.95, **kwargs):

    half_ci = ci / 2.0 * 100.0

    low_bound, hi_bound = np.percentile(
        y, [50.0 - half_ci, 50.0 + half_ci], axis=0
    )

    ax.fill_between(x, low_bound, hi_bound, **kwargs)
