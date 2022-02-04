import numpy as np
from matplotlib import colors


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


# 256 to [0,1]
def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


# [0,1] to 256
def infer_to_256(x):
    return int(np.interp(x=x, xp=[0, 1], fp=[0, 255]))


def build_custom_continuous_cmap(*rgb_list):
    """ """
    all_red = []
    all_green = []
    all_blue = []
    for rgb in rgb_list:
        all_red.append(rgb[0])
        all_green.append(rgb[1])
        all_blue.append(rgb[2])
    # build each section
    n_section = len(all_red) - 1
    red = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_red)
        ]
    )
    green = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_green)
        ]
    )
    blue = tuple(
        [
            (1 / n_section * i, inter_from_256(v), inter_from_256(v))
            for i, v in enumerate(all_blue)
        ]
    )
    cdict = {"red": red, "green": green, "blue": blue}
    new_cmap = colors.LinearSegmentedColormap("new_cmap", segmentdata=cdict)
    return new_cmap


def build_custom_divergent_cmap(hex_left, hex_right):
    """ """
    left_rgb = colors.to_rgb(hex_left)
    right_rgb = colors.to_rgb(hex_right)
    # build each section
    n_section = 2
    red = (
        (0, left_rgb[0], left_rgb[0]),
        (0.5, 1, 1),
        (1, right_rgb[0], right_rgb[0]),
    )
    green = (
        (0, left_rgb[1], left_rgb[1]),
        (0.5, 1, 1),
        (1, right_rgb[1], right_rgb[1]),
    )
    blue = (
        (0, left_rgb[2], left_rgb[2]),
        (0.5, 1, 1),
        (1, right_rgb[2], right_rgb[2]),
    )
    cdict = {"red": red, "green": green, "blue": blue}
    new_cmap = colors.LinearSegmentedColormap("new_cmap", segmentdata=cdict)
    return new_cmap
