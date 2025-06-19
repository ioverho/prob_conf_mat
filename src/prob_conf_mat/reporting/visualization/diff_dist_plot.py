from __future__ import annotations

import typing
import warnings

import numpy as np

from prob_conf_mat.experiment_comparison import pairwise_compare
from prob_conf_mat.utils import fmt

if typing.TYPE_CHECKING:
    from prob_conf_mat.metrics import Metric, AveragedMetric
    import matplotlib  # noqa: F401
    from matplotlib.figure import Figure

IMPLEMENTED_METHODS = {"kde", "hist", "histogram"}


def diff_distribution_plot(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    experiment_group_a: str,
    experiment_group_b: str,
    experiment_a: str | typing.Literal["aggregated"] = "aggregated",
    experiment_b: str | typing.Literal["aggregated"] = "aggregated",
    min_sig_diff: typing.Optional[float] = None,
    method: str = "kde",
    bandwidth: float = 1.0,
    bins: int | list[int] | str = "auto",
    figsize: typing.Optional[tuple[float, float]] = None,
    fontsize: float = 9,
    axis_fontsize: typing.Optional[float] = None,
    precision: int = 4,
    edge_colour="black",
    plot_min_sig_diff_lines: bool = True,
    min_sig_diff_lines_colour: str = "black",
    min_sig_diff_lines_format: str = "-",
    min_sig_diff_area_colour: str = "gray",
    min_sig_diff_area_alpha: float = 0.5,
    neg_sig_diff_area_colour: str = "red",
    neg_sig_diff_area_alpha: float = 0.5,
    pos_sig_diff_area_colour: str = "green",
    pos_sig_diff_area_alpha: float = 0.5,
    plot_obs_point: bool = True,
    obs_point_marker: str = "D",
    obs_point_colour: str = "black",
    obs_point_size: typing.Optional[float] = None,
    plot_median_line: bool = True,
    median_line_colour: str = "black",
    median_line_format: str = "--",
    plot_hdi_lines: bool = False,
    hdi_lines_colour: str = "black",
    hdi_lines_format: str = ":",
    plot_extrema_lines: bool = True,
    extrema_lines_colour: str = "black",
    extrema_lines_format: str = "-",
    extrema_line_height: float = 12,
    plot_base_line: bool = True,
    base_lines_colour: str = "black",
    base_lines_format: str = "-",
    plot_proportions: bool = True,
) -> Figure:
    """Plots the distribution of differences.

    Args:
        study (Study): the study object calling this function
        metric (Metric | AveragedMetric): the metric
        class_label (int): the class label to consider
        experiment_group_a (str): the name of the first experiment group
        experiment_group_b (str): the name of the second experiment group
        experiment_a (str | typing.Literal['aggregated']): the name of the first experiment. Defaults to 'aggregated'
        experiment_b (str | typing.Literal['aggregated']): the name of the first experiment. Defaults to 'aggregated'
        min_sig_diff (typing.Optional[float], optional): the minimum value for siginificance. Defaults to 0.1 * stdev.
        method (str, optional): the method for displaying a histogram, provided by Seaborn. Can be either a histogram or KDE. Defaults to "kde".
        bandwidth (float, optional): the bandwith parameter for the KDE. Corresponds to [Seaborn's `bw_adjust` parameter](https://seaborn.pydata.org/generated/seaborn.kdeplot.html). Defaults to 1.0.
        bins (int | list[int] | str, optional): the number of bins to use in the histrogram. Corresponds to [numpy's `bins` parameter](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges). Defaults to "auto".
        figsize (tuple[float, float], optional): the figure size, in inches. Corresponds to matplotlib's `figsize` parameter. Defaults to None, in which case a decent default value will be approximated.
        fontsize (float, optional): fontsize for the experiment name labels. Defaults to 9.
        axis_fontsize (float, optional): fontsize for the x-axis ticklabels. Defaults to None, in which case the fontsize will be used.
        edge_colour (str, optional): the colour of the histogram or KDE edge. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        plot_min_sig_diff_lines (bool, optional): whether to plot lines at the minimum significance values. Defaults to True.
        min_sig_diff_lines_colour (str, optional): the colour of the minimum significance lines. Defaults to "black".
        min_sig_diff_lines_format (str, optional): the format of the minimum significance lines. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        min_sig_diff_area_colour (str, optional): the colour of the region of practical equivalence (ROPE). Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "gray".
        min_sig_diff_area_alpha (float, optional): the opacity of the region of practical equivalence (ROPE). Corresponds to [matplotlib's `alpha` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#transparency). Defaults to 0.5.
        neg_sig_diff_area_colour (str, optional): the colour of the area in the significantly negative half. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "red".
        neg_sig_diff_area_alpha (float, optional): the opacity of the region in the significantly negative half. Corresponds to [matplotlib's `alpha` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#transparency). Defaults to 0.5.
        pos_sig_diff_area_colour (str, optional): the colour of the area in the significantly positive half. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "green".
        pos_sig_diff_area_alpha (float, optional): the opacity of the region in the significantly positive half. Corresponds to [matplotlib's `alpha` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#transparency). Defaults to 0.5.
        plot_obs_point (bool, optional): whether to plot the observed value as a marker. Defaults to True.
        obs_point_marker (str, optional): the marker type of the observed value. Corresponds to [matplotlib's `marker` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html#unfilled-markers). Defaults to "D".
        obs_point_colour (str, optional): the colour of the observed marker. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        obs_point_size (float, optional): the size of the observed marker. Defaults to None.
        plot_median_line (bool, optional): whether to plot the median line. Defaults to True.
        median_line_colour (str, optional): the colour of the median line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        median_line_format (str, optional): the format of the median line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "--".
        plot_extrema_lines (bool, optional): whether to plot small lines at the distribution extreme values. Defaults to True.
        extrema_lines_colour (str, optional): the colour of the extrema lines. Defaults to "black".
        extrema_lines_format (str, optional): the format of the extrema lines. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        extrema_line_height (int, optional): the height of the extremea lines in 'points' units. Defaults to 12.
        plot_base_line (bool, optional): whether to plot a line at the base of the distribution. Defaults to True.
        base_lines_colour (str, optional): the colour of the base line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        base_lines_format (str, optional): the format of the base line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        plot_proportions (bool, optional): whether to plot the proportions at the top of the figure. Defaults to True.
        precision (int, optional): the precision of the proportions. Defaults to 4.

    Returns:
        matplotlib.figure.Figure: the complete figure
    """

    try:
        # Import optional dependencies
        import matplotlib.pyplot as plt
        import seaborn as sns

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
        )

    diff_bounds = (
        metric.bounds[0] - metric.bounds[1],
        metric.bounds[1] - metric.bounds[0],
    )

    lhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_a,
        experiment=experiment_a,
        sampling_method="posterior",
    ).values[:, class_label]

    rhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_b,
        experiment=experiment_b,
        sampling_method="posterior",
    ).values[:, class_label]

    random_lhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_a,
        experiment=experiment_a,
        sampling_method="random",
    ).values[:, class_label]

    random_rhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_b,
        experiment=experiment_b,
        sampling_method="random",
    ).values[:, class_label]

    if experiment_a != "aggregated" and experiment_b != "aggregated":
        lhs_observed = study.get_metric_samples(
            metric=metric.name,
            experiment_group=experiment_group_a,
            experiment=experiment_a,
            sampling_method="input",
        ).values[:, class_label]

        rhs_observed = study.get_metric_samples(
            metric=metric.name,
            experiment_group=experiment_group_b,
            experiment=experiment_b,
            sampling_method="input",
        ).values[:, class_label]

        observed_diff = lhs_observed - rhs_observed
    else:
        observed_diff = None

    comparison_result = pairwise_compare(
        metric=metric,
        diff_dist=lhs_samples - rhs_samples,
        random_diff_dist=random_lhs_samples - random_rhs_samples,
        ci_probability=study.ci_probability,
        min_sig_diff=min_sig_diff,
        observed_difference=observed_diff,
    )

    # Figure instantiation
    if figsize is None:
        # Try to set a decent default figure size
        _figsize = (6.30, 2.52)
    else:
        _figsize = figsize

    fig, ax = plt.subplots(1, 1, figsize=_figsize)

    if method == "kde":
        # Plot the kde
        sns.kdeplot(
            comparison_result.diff_dist,
            ax=ax,
            fill=False,
            bw_adjust=bandwidth,
            color=edge_colour,
            clip=diff_bounds,
            zorder=2,
            clip_on=False,
        )

        kdeline = ax.lines[0]

        kde_x = kdeline.get_xdata()
        kde_y = kdeline.get_ydata()

    elif method == "hist" or method == "histogram":
        sns.histplot(
            comparison_result.diff_dist,
            fill=False,
            bins=bins,
            stat="density",
            element="step",
            ax=ax,
            color=edge_colour,
            zorder=2,
            clip_on=False,
        )

        kdeline = ax.lines[0]

        kde_x = kdeline.get_xdata()
        kde_y = kdeline.get_ydata()

        kde_x = np.repeat(kde_x, 2)
        kde_y = np.concatenate([[0], np.repeat(kde_y, 2)[:-1]])

    else:
        del fig, ax
        raise ValueError(
            f"Parameter `method` must be one of: {IMPLEMENTED_METHODS}. Currently: {method}"
        )

    min_x = np.min(kde_x)
    max_x = np.max(kde_x)

    if plot_min_sig_diff_lines:
        for msd in [-comparison_result.min_sig_diff, comparison_result.min_sig_diff]:
            y_msd = np.interp(
                x=msd,
                xp=kde_x,  # type: ignore
                fp=kde_y,  # type: ignore
            )  # type: ignore

            ax.vlines(
                msd,
                0,
                y_msd,
                color=min_sig_diff_lines_colour,
                linestyle=min_sig_diff_lines_format,
                zorder=1,
            )

    # Fill the ROPE
    rope_xx = np.linspace(
        -comparison_result.min_sig_diff,
        comparison_result.min_sig_diff,
        num=2 * kde_x.shape[0],  # type: ignore
    )

    rope_yy = np.interp(
        x=rope_xx,
        xp=kde_x,  # type: ignore
        fp=kde_y,  # type: ignore
    )  # type: ignore

    ax.fill_between(
        x=rope_xx,
        y1=0,
        y2=rope_yy,
        color=min_sig_diff_area_colour,
        alpha=min_sig_diff_area_alpha,
        interpolate=True,
        zorder=0,
        linewidth=0,
    )

    # Fill the negatively significant area
    neg_sig_xx = np.linspace(
        min_x,
        -comparison_result.min_sig_diff,
        num=2 * kde_x.shape[0],  # type: ignore
    )

    neg_sig_yy = np.interp(
        x=neg_sig_xx,
        xp=kde_x,  # type: ignore
        fp=kde_y,  # type: ignore
    )  # type: ignore

    ax.fill_between(
        x=neg_sig_xx,
        y1=0,
        y2=neg_sig_yy,
        color=neg_sig_diff_area_colour,
        alpha=neg_sig_diff_area_alpha,
        interpolate=True,
        zorder=0,
        linewidth=0,
    )

    # Fill the positively significant area
    pos_sig_xx = np.linspace(
        comparison_result.min_sig_diff,
        max_x,
        num=2 * kde_x.shape[0],  # type: ignore
    )

    pos_sig_yy = np.interp(  # type: ignore
        x=pos_sig_xx,
        xp=kde_x,  # type: ignore
        fp=kde_y,  # type: ignore
    )

    ax.fill_between(
        x=pos_sig_xx,
        y1=0,
        y2=pos_sig_yy,
        color=pos_sig_diff_area_colour,
        alpha=pos_sig_diff_area_alpha,
        interpolate=True,
        zorder=0,
        linewidth=0,
    )

    if plot_obs_point:
        # Add a point for the true point value
        observed_diff = comparison_result.observed_diff

        if observed_diff is not None:
            ax.scatter(
                observed_diff,
                0,
                marker=obs_point_marker,
                color=obs_point_colour,
                s=obs_point_size,
                clip_on=False,
                zorder=2,
            )
        else:
            warnings.warn(
                "Parameter `plot_obs_point` is True, but one of the experiments has no observation (i.e. aggregated). As a result, no observed difference will be shown."
            )

    if plot_median_line:
        # Plot median line
        median_x = comparison_result.diff_dist_summary.median

        y_median = np.interp(
            x=median_x,
            xp=kde_x,  # type: ignore
            fp=kde_y,  # type: ignore
        )  # type: ignore

        ax.vlines(
            median_x,
            0,
            y_median,
            color=median_line_colour,
            linestyle=median_line_format,
            zorder=1,
        )

        if plot_hdi_lines:
            x_hdi_lb = comparison_result.diff_dist_summary.hdi[0]

            y_hdi_lb = np.interp(
                x=x_hdi_lb,
                xp=kde_x,  # type: ignore
                fp=kde_y,  # type: ignore
            )  # type: ignore

            ax.vlines(
                x_hdi_lb,
                0,
                y_hdi_lb,
                color=hdi_lines_colour,
                linestyle=hdi_lines_format,
                zorder=1,
            )

            x_hdi_ub = comparison_result.diff_dist_summary.hdi[1]

            y_hdi_ub = np.interp(
                x=x_hdi_ub,
                xp=kde_x,  # type: ignore
                fp=kde_y,  # type: ignore
            )  # type: ignore

            ax.vlines(
                x_hdi_ub,
                0,
                y_hdi_ub,
                color=hdi_lines_colour,
                linestyle=hdi_lines_format,
                zorder=1,
            )

    if plot_base_line:
        # Add base line
        ax.hlines(
            0,
            min_x,
            max_x,
            clip_on=False,
            color=base_lines_colour,
            ls=base_lines_format,
            zorder=3,
        )

    if plot_extrema_lines:
        standard_length = (
            ax.transData.inverted().transform([0, extrema_line_height])[1]
            - ax.transData.inverted().transform([0, 0])[1]
        )

        # Add lines for the horizontal extrema
        ax.vlines(
            min_x,
            0,
            standard_length,
            clip_on=False,
            color=extrema_lines_colour,
            ls=extrema_lines_format,
            zorder=3,
        )
        ax.vlines(
            max_x,
            0,
            standard_length,
            clip_on=False,
            color=extrema_lines_colour,
            ls=extrema_lines_format,
            zorder=3,
        )

    # Add text labels for the proportion in the different regions
    cur_ylim = ax.get_ylim()
    cur_xlim = ax.get_xlim()

    if plot_proportions:
        if max_x > comparison_result.min_sig_diff:
            # The proportion in the positively significant region
            ax.text(
                s=f"$p_{{sig}}^{{+}}$\n{fmt(comparison_result.p_sig_pos, precision=precision, mode='%')}\n",
                x=0.5 * (cur_xlim[1] + comparison_result.min_sig_diff),
                y=cur_ylim[1],
                horizontalalignment="center",
                verticalalignment="center_baseline",
                fontsize=fontsize,
                color=pos_sig_diff_area_colour,
            )

        if min_x < -comparison_result.min_sig_diff:
            # The proportion in the negatively significant area
            ax.text(
                s=f"$p_{{sig}}^{{-}}$\n{fmt(comparison_result.p_sig_neg, precision=precision, mode='%')}\n",
                x=0.5 * (cur_xlim[0] - comparison_result.min_sig_diff),
                y=cur_ylim[1],
                horizontalalignment="center",
                verticalalignment="center_baseline",
                fontsize=fontsize,
                color=neg_sig_diff_area_colour,
            )

        # The proportion in the ROPE
        ax.text(
            s=f"$p_{{sig}}^{{-}}$\n{fmt(comparison_result.p_rope, precision=precision, mode='%')}\n",
            x=0.0,
            y=cur_ylim[1],
            horizontalalignment="center",
            verticalalignment="center_baseline",
            fontsize=fontsize,
            color=min_sig_diff_area_colour,
        )

    # Remove the y ticks
    ax.set_yticks([])
    ax.set_ylabel("")

    # Remove the axis spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlim(
        min(-comparison_result.min_sig_diff, cur_xlim[0]),
        max(cur_xlim[1], comparison_result.min_sig_diff),
    )

    ax.tick_params(
        axis="x", labelsize=axis_fontsize if axis_fontsize is not None else fontsize
    )

    fig.tight_layout()

    return fig
