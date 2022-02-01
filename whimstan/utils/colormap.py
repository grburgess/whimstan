from matplotlib import cm
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb, to_rgba
from matplotlib import colors
import copy
import matplotlib.pyplot as plt


# ASCI escape codes
def color_stdout(skk, c):
    """
    color your output to the terminal
    :param skk: the string you want to color
    :param c: the name of the color 'red','green','yellow','lightpurple','cyan','lightgrey','black'
    Example::
        from color import color_stdout
        # when print to terminal, it will be red
        print(color_stdout('hello','red'))
    """
    if c == "red":
        return "\033[91m {}\033[00m".format(skk)
    elif c == "green":
        return "\033[92m {}\033[00m".format(skk)
    elif c == "yellow":
        return "\033[93m {}\033[00m".format(skk)
    elif c == "lightpurple":
        return "\033[94m {}\033[00m".format(skk)
    elif c == "cyan":
        return "\033[95m {}\033[00m".format(skk)
    elif c == "lightgrey":
        return "\033[97m {}\033[00m".format(skk)
    elif c == "black":
        return "\033[98m {}\033[00m".format(skk)


# test_discrete_look
def generate_block(color_list, name):
    """
    Given a list of color (each item is a hex code), visualize them side by side. See example.
    """
    n = len(color_list)
    strip = np.empty(shape=(1, 256), dtype="<U7")
    splitted = np.array_split(np.arange(strip.shape[1]), n)
    for i, c in enumerate(color_list):
        strip[:, splitted[i]] = c
    block = np.repeat(strip, 10, axis=0)
    block_rgb = hex2_to_rgb3(block)
    fig, ax = plt.subplots()
    ax.imshow(block_rgb)
    ax.axis("off")
    ax.set_title("{}".format(name))
    plt.savefig("{}_block.pdf".format(name), bbox_inches="tight")
    plt.close()


# test_cmap_look
def generate_gradient(cmap, name):
    """
    Given a continuous cmap, visualize them. See example.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.repeat(gradient, 10, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(gradient, cmap=cmap)
    ax.axis("off")
    ax.set_title("{}".format(name))
    plt.savefig("{}_gradient.pdf".format(name), bbox_inches="tight")
    plt.close()


# background greyed colormap
def bg_greyed_cmap(cmap_str):
    """
    set 0 value as lightgrey, which will render better effect on umap
    :param cmap_str: string, any valid matplotlib colormap string
    :return: colormap object
    Examples::
        # normal cmap
        sc.pl.umap(sctri.adata,color='CD4',cmap='viridis')
        plt.savefig('normal.pdf',bbox_inches='tight')
        plt.close()
        # bg_greyed cmap
        sc.pl.umap(sctri.adata,color='CD4',cmap=bg_greyed_cmap('viridis'),vmin=1e-5)
        plt.savefig('bg_greyed.pdf',bbox_inches='tight')
        plt.close()
    .. image:: ./_static/normal.png
        :height: 300px
        :width: 300px
        :align: left
        :target: target
    .. image:: ./_static/bg_greyed.png
        :height: 300px
        :width: 300px
        :align: right
        :target: target
    """
    # give a matplotlib cmap str, for instance, 'viridis' or 'YlOrRd'
    cmap = copy.copy(cm.get_cmap(cmap_str))
    cmap.set_under("lightgrey")
    return cmap


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

# hex color 2d array, to (M,N,3) RGB array, used in imshow (plot_long_heatmap)
def hex2_to_rgb3(hex2):
    """
    convert a hex color 2d array to (M,N,3) RGB array, very useful in ``ax.imshow``
    """
    rgb3 = np.empty([hex2.shape[0], hex2.shape[1], 3])
    for i in range(hex2.shape[0]):
        for j in range(hex2.shape[1]):
            hex_ = hex2[i][j]
            rgb_ = to_rgb(hex_)
            rgb3[i, j, :] = rgb_
    return rgb3


# 256 to [0,1]
def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


# [0,1] to 256
def infer_to_256(x):
    return int(np.interp(x=x, xp=[0, 1], fp=[0, 255]))


def build_custom_continuous_cmap(*rgb_list):
    """
    Generating any custom continuous colormap, user should supply a list of (R,G,B) color taking the value from [0,255], because this is
    the format the adobe color will output for you.
    Examples::
        test_cmap = build_custom_continuous_cmap([64,57,144],[112,198,162],[230,241,146],[253,219,127],[244,109,69],[169,23,69])
        fig,ax = plt.subplots()
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=diverge_cmap),ax=ax)
    .. image:: ./_static/custom_continuous_cmap.png
        :height: 400px
        :width: 550px
        :align: center
        :target: target
    """
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
    """
    User supplies two arbitrary hex code for the vmin and vmax color values, then it will build a divergent cmap centers at pure white.
    Examples::
        diverge_cmap = build_custom_divergent_cmap('#21EBDB','#F0AA5F')
        fig,ax = plt.subplots()
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(),cmap=diverge_cmap),ax=ax)
    .. image:: ./_static/custom_divergent_cmap.png
        :height: 400px
        :width: 550px
        :align: center
        :target: target
    """
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
