import typing

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jaxtyping as jtyping
import numpy as np

from bayes_conf_mat.report.utils.stats import hdi_estimator

MAX_HIST_HEIGHT = 0.75


def forest_plot(
    individual_samples: typing.List[
        typing.Tuple[str, jtyping.Float[np.ndarray, " num_samples"]]
    ],
    aggregated_samples: typing.List[
        typing.Tuple[str, jtyping.Float[np.ndarray, " num_samples"]]
    ],
    extrema: typing.Tuple[float, float],
    hdi_prob: float,
    agg_offset: typing.Optional[int] = 1,
    fontsize: typing.Optional[int] = 10,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
):
    # Try to automatically determine figsize
    if figsize is None:
        figsize = [None, None]
        figsize[0] = 6.29921
        figsize[1] = (len(individual_samples) + 2 + agg_offset) / 2.5

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    # Plot the individual experiments
    smallest_min = float("inf")
    largest_max = -float("inf")
    y_ticklabels_left = []
    y_ticklabels_right = []
    for i, (experiment_name, experiment_samples) in enumerate(individual_samples):
        median = np.median(experiment_samples)
        hdi = hdi_estimator(experiment_samples, prob=hdi_prob)
        uncertainty = hdi[1] - hdi[0]

        y_ticklabels_left.append(experiment_name)
        y_ticklabels_right.append([median, *hdi, uncertainty])

        ax.scatter(
            x=median,
            y=i,
            marker="s",
            facecolor="white",
            s=7**2,
            edgecolor="black",
            linewidth=1.5,
            zorder=1,
        )

        ax.hlines(
            xmin=hdi[0],
            xmax=hdi[1],
            y=i,
            zorder=0,
            # linewidth=0.005 * pts,
            color="black",
        )

        if hdi[0] < smallest_min:
            smallest_min = hdi[0]

        if hdi[1] > largest_max:
            largest_max = hdi[1]

        counts, bins = np.histogram(
            experiment_samples,
            bins="auto",
        )

        counts = counts / counts.max()

        for l_edge, r_edge, count in zip(bins[:-1], bins[1:], counts):
            bar = patches.Rectangle(
                xy=(l_edge, i - MAX_HIST_HEIGHT * count),
                height=MAX_HIST_HEIGHT * count,
                width=l_edge - r_edge,
                color="black",
                alpha=0.30,
                linewidth=0.0,
                # linewidth=1,
                zorder=-1,
            )
            ax.add_patch(bar)

    agg_base = i + 1 + agg_offset
    first_agg_median = None

    central_point = 0.0
    for ii, (agg_name, agg_samples) in enumerate(aggregated_samples):
        # Plot the aggregated measures
        # Handle the conflated/aggregated/overall distribution
        median = np.median(agg_samples)
        if ii == 0:
            first_agg_median = median
        else:
            ax.vlines(
                ymin=agg_base,
                ymax=agg_base + ii,
                x=median,
                zorder=-1,
                colors="black",
                linestyles="dotted",
            )

        hdi = hdi_estimator(agg_samples, prob=hdi_prob)

        uncertainty = hdi[1] - hdi[0]

        y_ticklabels_left.append(agg_name)
        y_ticklabels_right.append([median, *hdi, uncertainty])

        central_point += median

        # Median
        ax.scatter(
            x=median,
            y=agg_base + ii,
            marker="D",
            facecolor="white",
            s=8**2,
            edgecolor="black",
            linewidth=1.5,
            zorder=1,
        )

        # HDI
        ax.hlines(
            xmin=hdi[0],
            xmax=hdi[1],
            y=agg_base + ii,
            zorder=0,
            color="black",
        )

        # Distribution
        counts, bins = np.histogram(
            agg_samples,
            bins="auto",
        )

        counts = counts / counts.max()

        for l_edge, r_edge, count in zip(bins[:-1], bins[1:], counts):
            bar = patches.Rectangle(
                xy=(l_edge, agg_base + ii - MAX_HIST_HEIGHT * count),
                height=MAX_HIST_HEIGHT * count,
                width=l_edge - r_edge,
                color="black",
                alpha=0.30,
                linewidth=0.0,
                # linewidth=1,
                zorder=0,
            )
            ax.add_patch(bar)

    # Plot the median line
    # Without disturbing plot y-extremes
    ax_ylim = ax.get_ylim()

    ax.vlines(
        ymin=ax_ylim[0],
        ymax=agg_base,
        x=first_agg_median,
        zorder=-1,
        colors="black",
        linestyles="dashed",
    )

    # Provide experiment labels
    ax.set_yticks(
        [ax_ylim[0]]
        + [i for i in range(len(individual_samples))]
        + [agg_base + ii for ii in range(len(aggregated_samples))]
    )

    longest_experiment_name = max(len(label) for label in y_ticklabels_left)
    experiment_name_size = max(15, longest_experiment_name)

    ax.set_yticklabels(
        [f"{'Experiment Name':<{experiment_name_size}}{'':5}"]
        + [
            f"{experiment_name[:experiment_name_size]:<{experiment_name_size}}{'':5}"
            for experiment_name in y_ticklabels_left
        ],
        fontsize=fontsize,
        fontname="monospace",
    )

    # Remove the y-ticks
    ax.tick_params(axis="y", which="both", length=0)

    # Invert the y-axes
    ax.set_ylim(ax_ylim[::-1])
    # ax_clone.set_ylim(ax_ylim[::-1])

    # Set the x-ticks
    central_point = central_point / len(aggregated_samples)
    largest_deviation = max(largest_max - central_point, central_point - smallest_min)

    plot_range_min = max(
        extrema[0] - 0.05,
        central_point - 1.25 * largest_deviation,
    )

    plot_range_max = min(
        extrema[1] + 0.05,
        central_point + 1.25 * largest_deviation,
    )

    ax.set_xlim(
        plot_range_min,
        plot_range_max,
    )

    # Remove the spines from the forest plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="x", which="both", labelsize=fontsize)

    # Add summary info to right side of plot
    ax_clone = ax.twinx()

    ax_clone.set_yticks(
        [ax_ylim[0]]
        + [i for i in range(len(individual_samples))]
        + [agg_base + ii for ii in range(len(aggregated_samples))]
    )

    ax_clone.set_yticklabels(
        [f"{'':5}{'Median':^6}{'':3}{f'{hdi_prob*100:.2f}%HDI':^16}{'':3}{'Uncert':^6}"]
        + [
            f"{'':5}{median:.4f}{'':3}[{hdi_lb:.4f}, {hdi_ub:.4f}]{'':3}{uncertainty:.4f}"
            for (median, hdi_lb, hdi_ub, uncertainty) in y_ticklabels_right
        ],
        fontsize=fontsize,
        fontname="monospace",
    )

    ax_clone.spines["right"].set_visible(False)
    ax_clone.spines["top"].set_visible(False)
    ax_clone.spines["left"].set_visible(False)
    ax_clone.tick_params(axis="x", which="both", labelsize=fontsize)
    ax_clone.tick_params(axis="y", which="both", length=0)
    ax_clone.set_ylim(ax_ylim[::-1])

    fig.subplots_adjust()
    fig.tight_layout()

    return fig
