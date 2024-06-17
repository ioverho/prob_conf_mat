import typing

import matplotlib.pyplot as plt

from bayes_conf_mat.significance_testing import PairwiseComparisonResult
from bayes_conf_mat.utils.formatting import fmt


POS_REGION_COLOUR = "#1E88E5"
NEUTRAL_REGION_COLOUR = "#797979"
NEG_REGION_COLOUR = "#D81B60"


def hex_to_rgb(hex_colour: str, norm: bool = False, rgba: float = None):
    hex_colour = hex_colour.lstrip("#")

    rgb_colour = []
    for i in (0, 2, 4):
        base_10 = int(hex_colour[i : i + 2], 16)
        if norm:
            rgb_colour.append(base_10 / 255)
        else:
            rgb_colour.append(base_10 / 255)

    if rgba is not None:
        rgb_colour.append(rgba)

    return rgb_colour


def pairwise_comparison_plot(
    result: PairwiseComparisonResult,
    figsize: typing.Optional[typing.Tuple[float, float]] = None,
    precision: int = 4,
    fontsize: int = 11,
):
    # Try to set a reasonable figsize
    if figsize is None:
        figsize = (6.29921, 1.787402)

    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=figsize,
    )

    # Determine the horizontal range to plot
    min_val, max_val = (
        min(-result.min_sig_diff, result.diff_dist.min()),
        max(result.min_sig_diff, result.diff_dist.max()),
    )
    ax.set_xlim(min_val, max_val)

    # Remove yticks
    ax.set(yticks=[], ylabel="", xlabel="")
    ax.tick_params(axis="x", labelsize=fontsize)

    # Plot the distribution's density
    ax.hist(result.diff_dist, zorder=2, bins="auto")

    # Colour the different regions
    # Tries to find the minimal set of columns that encompass the ROPE
    for child in ax._children:
        child_x, _ = child.get_xy()

        if child_x > result.min_sig_diff:
            child.set_facecolor(hex_to_rgb(POS_REGION_COLOUR, norm=True, rgba=1.0))
        elif child_x + child.get_width() < -result.min_sig_diff:
            child.set_facecolor(hex_to_rgb(NEG_REGION_COLOUR, norm=True, rgba=1.0))
        else:
            child.set_facecolor(hex_to_rgb(NEUTRAL_REGION_COLOUR, norm=True, rgba=1.0))

    cur_ylim = ax.get_ylim()

    # Plot the origin line
    ax.vlines(
        x=0.0,
        ymin=cur_ylim[0],
        ymax=cur_ylim[1],
        linestyles="dashed",
        colors="black",
        zorder=3,
    )

    # Add a light gray background to te insignificant section
    ax.fill_between(
        x=[-result.min_sig_diff, result.min_sig_diff],
        y1=cur_ylim[0],
        y2=cur_ylim[1],
        color=hex_to_rgb(
            NEUTRAL_REGION_COLOUR,
            norm=True,
            rgba=0.25,
        ),
        edgecolor=None,
        zorder=1,
    )

    # Add text labels for the proportion in the different regions
    cur_xlim = ax.get_xlim()

    # The proportion in the positive region
    ax.text(
        s=f"$p_{{sig}}^{{+}}$\n{fmt(result.p_sig_pos, precision=precision, mode='%')}\n",
        x=0.5 * (cur_xlim[1] + result.min_sig_diff),
        y=cur_ylim[1],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=fontsize,
        color=POS_REGION_COLOUR,
    )

    # The proportion in the negative region
    ax.text(
        s=f"$p_{{sig}}^{{-}}$\n{fmt(result.p_sig_neg, precision=precision, mode='%')}\n",
        x=0.5 * (cur_xlim[0] - result.min_sig_diff),
        y=cur_ylim[1],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=fontsize,
        color=NEG_REGION_COLOUR,
    )

    # The proportion in the ROPE
    ax.text(
        s=f"$p_{{insig}}$\n{fmt(result.p_rope, precision=precision, mode='%')}\n",
        x=0.0,
        y=cur_ylim[1],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=fontsize,
        color="black",
    )

    fig.tight_layout()

    return fig
