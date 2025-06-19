from __future__ import annotations

import typing

import numpy as np

from prob_conf_mat.stats import summarize_posterior

if typing.TYPE_CHECKING:
    from prob_conf_mat.metrics import Metric, AveragedMetric
    import matplotlib  # noqa: F401
    from matplotlib.figure import Figure

IMPLEMENTED_METHODS = {"kde", "hist", "histogram"}


def distribution_plot(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    method: str = "kde",
    bandwidth: float = 1.0,
    bins: int | list[int] | str = "auto",
    normalize: bool = False,
    figsize: typing.Optional[tuple[float, float]] = None,
    fontsize: float = 9,
    axis_fontsize: typing.Optional[float] = None,
    edge_colour: str = "black",
    area_colour: str = "gray",
    area_alpha: float = 0.5,
    plot_median_line: bool = True,
    median_line_colour: str = "black",
    median_line_format: str = "--",
    plot_hdi_lines: bool = True,
    hdi_lines_colour: str = "black",
    hdi_line_format: str = "-",
    plot_obs_point: bool = True,
    obs_point_marker: str = "D",
    obs_point_colour: str = "black",
    obs_point_size: typing.Optional[float] = None,
    plot_extrema_lines: bool = True,
    extrema_lines_colour: str = "black",
    extrema_lines_format: str = "-",
    extrema_line_height: float = 12,
    extrema_lines_width: float = 1,
    plot_base_line: bool = True,
    base_line_colour: str = "black",
    base_line_format: str = "-",
    base_line_width: int = 1,
    plot_experiment_name: bool = True,
) -> Figure:
    """Plots a distribution.

    Args:
        study (Study): the study object calling this function
        metric (Metric | AveragedMetric): the metric
        method (str, optional): the method for displaying a histogram, provided by Seaborn. Can be either a histogram or KDE. Defaults to "kde".
        bandwidth (float, optional): the bandwith parameter for the KDE. Corresponds to [Seaborn's `bw_adjust` parameter](https://seaborn.pydata.org/generated/seaborn.kdeplot.html). Defaults to 1.0.
        bins (int | list[int] | str, optional): the number of bins to use in the histrogram. Corresponds to [numpy's `bins` parameter](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges). Defaults to "auto".
        normalize (bool, optional): if normalized, each distribution will be scaled to [0, 1]. Otherwise, uses a shared y-axis. Defaults to False.
        figsize (tuple[float, float], optional): the figure size, in inches. Corresponds to matplotlib's `figsize` parameter. Defaults to None, in which case a decent default value will be approximated.
        fontsize (float, optional): fontsize for the experiment name labels. Defaults to 9.
        axis_fontsize (float, optional): fontsize for the x-axis ticklabels. Defaults to None, in which case the fontsize will be used.
        edge_colour (str, optional): the colour of the histogram or KDE edge. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        area_colour (str, optional): the colour of the histogram or KDE filled area. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "gray".
        area_alpha (float, optional): the opacity of the histogram or KDE filled area. Corresponds to [matplotlib's `alpha` parameter](). Defaults to 0.5.
        plot_median_line (bool, optional): whether to plot the median line. Defaults to True.
        median_line_colour (str, optional): the colour of the median line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        median_line_format (str, optional): the format of the median line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "--".
        plot_hdi_lines (bool, optional): whether to plot the HDI lines. Defaults to True.
        hdi_lines_colour (str, optional): the colour of the HDI lines. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        hdi_line_format (str, optional): the format of the HDI lines. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        plot_obs_point (bool, optional): whether to plot the observed value as a marker. Defaults to True.
        obs_point_marker (str, optional): the marker type of the observed value. Corresponds to [matplotlib's `marker` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html#unfilled-markers). Defaults to "D".
        obs_point_colour (str, optional): the colour of the observed marker. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        obs_point_size (float, optional): the size of the observed marker. Defaults to None.
        plot_extrema_lines (bool, optional): whether to plot small lines at the distribution extreme values. Defaults to True.
        extrema_lines_colour (str, optional): the colour of the extrema lines. Defaults to "black".
        extrema_lines_format (str, optional): the format of the extrema lines. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        extrema_line_height (int, optional): the height of the extremea lines in 'points' units. Defaults to 8.
        plot_base_line (bool, optional): whether to plot a line at the base of the distribution. Defaults to True.
        base_line_colour (str, optional): the colour of the base line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
        base_line_format (str, optional): the format of the base line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
        plot_experiment_name (bool, optional): whether to plot the experiment names as labels. Defaults to True.

    Returns:
        matplotlib.figure.Figure: the complete figure
    """

    try:
        # Import optional dependencies
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
        )

    total_num_experiments = sum(map(len, study.experiment_groups.values()))

    if figsize is None:
        # Try to set a decent default figure size
        _figsize = (6.29921, max(0.625 * total_num_experiments, 2.5))
    else:
        _figsize = figsize

    fig, axes = plt.subplots(
        total_num_experiments, 1, figsize=_figsize, sharey=(not normalize)
    )

    if total_num_experiments == 1:
        axes = np.array([axes])

    metric_bounds = metric.bounds

    i = 0

    all_min_x = []
    all_max_x = []
    all_max_height = []
    all_medians = []
    all_hdi_ranges = []
    for experiment_group_name, experiment_group in study.experiment_groups.items():
        for experiment_name, experiment in experiment_group.experiments.items():
            if plot_experiment_name:
                # Set the axis title
                # Needs to happen before KDE
                axes[i].set_ylabel(
                    f"{experiment_group_name}/{experiment_name}",
                    rotation=0,
                    va="center",
                    ha="right",
                    fontsize=fontsize,
                )

            distribution_samples = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="posterior",
            ).values[:, class_label]

            # Get summary statistics
            posterior_summary = summarize_posterior(
                distribution_samples, ci_probability=study.ci_probability
            )

            all_medians.append(posterior_summary.median)
            all_hdi_ranges.append(posterior_summary.hdi[1] - posterior_summary.hdi[0])

            if method not in IMPLEMENTED_METHODS:
                del fig, axes
                raise ValueError(
                    f"Parameter `method` must be one of: {IMPLEMENTED_METHODS}. Currently: {method}"
                )

            elif method == "kde":
                # Plot the kde
                sns.kdeplot(
                    distribution_samples,
                    fill=False,
                    bw_adjust=bandwidth,
                    ax=axes[i],
                    color=edge_colour,
                    clip=metric_bounds,
                    zorder=2,
                )

                kdeline = axes[i].lines[0]

                kde_x = kdeline.get_xdata()
                kde_y = kdeline.get_ydata()

                all_min_x.append(kde_x[0])
                all_max_x.append(kde_x[-1])
                all_max_height.append(np.max(kde_y))

                if area_colour is not None:
                    axes[i].fill_between(
                        kde_x, kde_y, color=area_colour, zorder=0, alpha=area_alpha
                    )

            elif method == "hist" or method == "histogram":
                sns.histplot(
                    distribution_samples,
                    fill=False,
                    bins=bins,
                    stat="density",
                    element="step",
                    ax=axes[i],
                    color=edge_colour,
                    zorder=2,
                )

                kdeline = axes[i].lines[0]

                kde_x = kdeline.get_xdata()
                kde_y = kdeline.get_ydata()

                all_min_x.append(kde_x[0])
                all_max_x.append(kde_x[-1])
                all_max_height.append(np.max(kde_y))

                kde_x = np.repeat(kde_x, 2)
                kde_y = np.concatenate([[0], np.repeat(kde_y, 2)[:-1]])

                if area_colour is not None:
                    axes[i].fill_between(
                        kde_x,
                        kde_y,
                        color=area_colour,
                        zorder=0,
                        alpha=area_alpha,
                    )

            if plot_obs_point:
                # Add a point for the true point value
                observed_metric_value = study.get_metric_samples(
                    metric=metric.name,
                    experiment_group=experiment_group_name,
                    experiment=experiment_name,
                    sampling_method="input",
                ).values[:, class_label]

                axes[i].scatter(
                    observed_metric_value,
                    0,
                    marker=obs_point_marker,
                    color=obs_point_colour,
                    s=obs_point_size,
                    clip_on=False,
                    zorder=2,
                )

            if plot_median_line:
                # Plot median line
                median_x = posterior_summary.median

                y_median = np.interp(
                    x=median_x,
                    xp=kde_x,  # type: ignore
                    fp=kde_y,  # type: ignore
                )  # type: ignore

                axes[i].vlines(
                    median_x,
                    0,
                    y_median,
                    color=median_line_colour,
                    linestyle=median_line_format,
                    zorder=1,
                )

            if plot_hdi_lines:
                x_hdi_lb = posterior_summary.hdi[0]

                y_hdi_lb = np.interp(
                    x=x_hdi_lb,
                    xp=kde_x,  # type: ignore
                    fp=kde_y,  # type: ignore
                )  # type: ignore

                axes[i].vlines(
                    x_hdi_lb,
                    0,
                    y_hdi_lb,
                    color=hdi_lines_colour,
                    linestyle=hdi_line_format,
                    zorder=1,
                )

                x_hdi_ub = posterior_summary.hdi[1]

                y_hdi_ub = np.interp(
                    x=x_hdi_ub,
                    xp=kde_x,  # type: ignore
                    fp=kde_y,  # type: ignore
                )  # type: ignore

                axes[i].vlines(
                    x_hdi_ub,
                    0,
                    y_hdi_ub,
                    color=hdi_lines_colour,
                    linestyle=hdi_line_format,
                    zorder=1,
                )

            i += 1

    smallest_hdi_range_i = int(np.argmin(all_hdi_ranges))

    grand_min_x = max(
        np.min(all_min_x),
        all_medians[smallest_hdi_range_i] - 5 * all_hdi_ranges[smallest_hdi_range_i],
    )
    grand_max_x = min(
        np.max(all_max_x),
        all_medians[smallest_hdi_range_i] + 5 * all_hdi_ranges[smallest_hdi_range_i],
    )

    # Decide on the xlim
    data_range = grand_min_x - grand_max_x
    metric_range = metric_bounds[1] - metric_bounds[0]

    # If the data range spans more than half the metric range
    # Just plot the whole metric range
    if (
        data_range / metric_range > 0.5
        and np.isfinite(metric_bounds[0])
        and np.isfinite(metric_bounds[1])
    ):
        x_lim_min = metric_bounds[0]
        x_lim_max = metric_bounds[1]
    else:
        # If close enough to the metric minimum, use that value
        if (
            np.isfinite(metric_range)
            and (grand_min_x - metric_bounds[0]) / metric_range < 0.05
        ):
            x_lim_min = metric_bounds[0]
        else:
            x_lim_min = grand_min_x  # - 0.05 * (grand_max_x - grand_min_x)

        # If close enough to the metric maximum, use that value
        if (
            np.isfinite(metric_range)
            and (metric_bounds[1] - grand_max_x) / metric_range < 0.05
        ):
            x_lim_max = metric_bounds[1]
        else:
            x_lim_max = grand_max_x  # + 0.05 * (grand_max_x - grand_min_x)

    for ax in axes:
        ax.set_xlim(x_lim_min, x_lim_max)

    for i, ax in enumerate(axes):
        if plot_base_line:
            # Add base line
            ax.hlines(
                0,
                max(all_min_x[i], grand_min_x),
                min(all_max_x[i], grand_max_x),
                color=base_line_colour,
                ls=base_line_format,
                linewidth=base_line_width,
                zorder=3,
                clip_on=False,
            )

        standard_length = (
            ax.transData.inverted().transform([0, extrema_line_height])[1]
            - ax.transData.inverted().transform([0, 0])[1]
        )

        if plot_extrema_lines:
            # Add lines for the horizontal extrema
            if all_min_x[i] >= grand_min_x:
                ax.vlines(
                    all_min_x[i],
                    0,
                    standard_length,
                    color=extrema_lines_colour,
                    ls=extrema_lines_format,
                    linewidth=extrema_lines_width,
                    zorder=3,
                    clip_on=False,
                )

            if all_max_x[i] <= grand_max_x:
                ax.vlines(
                    all_max_x[i],
                    0,
                    standard_length,
                    color=extrema_lines_colour,
                    ls=extrema_lines_format,
                    zorder=3,
                    linewidth=extrema_lines_width,
                    clip_on=False,
                )

        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove the axis spine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Add the axes back, but only for the bottom plot
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].xaxis.set_major_locator(matplotlib.ticker.AutoLocator())  # type: ignore
    axes[-1].set_yticks([])
    axes[-1].tick_params(
        axis="x", labelsize=axis_fontsize if axis_fontsize is not None else fontsize
    )

    fig.tight_layout()

    return fig
