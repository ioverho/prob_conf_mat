import typing

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from bayes_conf_mat.experiment_manager import (
    SplitExperimentResult,
    ExperimentAggregationResult,
)
from bayes_conf_mat.stats import hdi_estimator
from bayes_conf_mat.utils.formatting import fmt


# TODO: document this function
def forest_plot(
    individual_samples: typing.List[SplitExperimentResult],
    aggregated_samples: typing.List[ExperimentAggregationResult],
    bounds: typing.Tuple[float, float],
    ci_probability: float,
    precision: int = 4,
    fontsize: typing.Optional[int] = 9,
    figsize: typing.Optional[typing.Tuple[int, int]] = None,
    add_summary_info: bool = True,
    agg_offset: typing.Optional[int] = 1,
    max_hist_height: float = 0.7,
):
    # Try to automatically determine figsize
    if figsize is None:
        figsize = [None, None]
        figsize[0] = 6.29921
        figsize[1] = (len(individual_samples) + 2 + agg_offset) / 2.5

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    # Plot the individual experiments
    individ_bins = []
    individ_counts = []
    smallest_min = float("inf")
    largest_max = -float("inf")
    y_ticklabels_left = []
    y_ticklabels_right = []
    for i, experiment_result in enumerate(individual_samples):
        median = np.median(experiment_result.values)
        hdi = hdi_estimator(experiment_result.values, prob=ci_probability)
        uncertainty = hdi[1] - hdi[0]

        y_ticklabels_left.append(experiment_result.experiment.name)
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
            experiment_result.values,
            bins="auto",
            density=True,
        )

        individ_bins.append(bins)
        individ_counts.append(counts)

    agg_base = i + 1 + agg_offset
    first_agg_median = None

    agg_bins = []
    agg_counts = []
    central_point = 0.0
    for ii, agg_result in enumerate(aggregated_samples):
        # Plot the aggregated measures
        # Handle the conflated/aggregated/overall distribution
        median = np.median(agg_result.values)
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

        hdi = hdi_estimator(agg_result.values, prob=ci_probability)

        uncertainty = hdi[1] - hdi[0]

        y_ticklabels_left.append(agg_result.aggregator.name)
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
            agg_result.values,
            bins="auto",
            density=True,
        )

        agg_bins.append(bins)
        agg_counts.append(counts)

    # Actually plot the histograms
    # Divide the individual counts by the global maximum count
    highest_count = max(max(counts) for counts in individ_counts)
    for i, (bins, counts) in enumerate(zip(individ_bins, individ_counts)):
        for l_edge, r_edge, count in zip(bins[:-1], bins[1:], counts):
            bar = patches.Rectangle(
                xy=(l_edge, i - max_hist_height * count / highest_count),
                height=max_hist_height * count / highest_count,
                width=l_edge - r_edge,
                color="black",
                alpha=0.30,
                linewidth=0.0,
                # linewidth=1,
                zorder=-1,
            )
            ax.add_patch(bar)

    for ii, (bins, counts) in enumerate(zip(agg_bins, agg_counts)):
        for l_edge, r_edge, count in zip(bins[:-1], bins[1:], counts):
            bar = patches.Rectangle(
                xy=(
                    l_edge,
                    agg_base + ii - max_hist_height * count / highest_count,
                ),
                height=max_hist_height * count / highest_count,
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
        [f"{'Experiment Name':<{experiment_name_size}}{'':2}"]
        + [
            f"{experiment_name[:experiment_name_size]:<{experiment_name_size}}{'':2}"
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
        bounds[0],
        central_point - 1.25 * largest_deviation,
    )

    plot_range_max = min(
        bounds[1],
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

    if add_summary_info:
        # Add summary info to right side of plot
        ax_clone = ax.twinx()

        ax_clone.set_yticks(
            [ax_ylim[0]]
            + [i for i in range(len(individual_samples))]
            + [agg_base + ii for ii in range(len(aggregated_samples))]
        )

        ax_clone.set_yticklabels(
            [
                f"{'':1}{'Median':^6}{'':1}{f'{ci_probability*100:.2f}%HDI':^16}{'':1}{'MU':^6}"
            ]
            + [
                f"{'':1}{fmt(median, precision=precision, mode='f')}{'':1}[{fmt(hdi_lb, precision=precision, mode='f')}, {fmt(hdi_ub, precision=precision, mode='f')}]{'':1}{fmt(uncertainty, precision=precision, mode='f')}"
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
