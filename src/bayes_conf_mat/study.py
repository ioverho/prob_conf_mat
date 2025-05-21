from __future__ import annotations
import typing
import warnings
from collections import OrderedDict
from functools import cache
from enum import StrEnum

from bayes_conf_mat.experiment_comparison.pairwise import PairwiseComparisonResult

if typing.TYPE_CHECKING:
    from bayes_conf_mat.utils.typing import MetricLike
    import matplotlib
    import matplotlib.figure
    import matplotlib.axes
    import matplotlib.ticker

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.config import Config, ConfigError, ConfigWarning
from bayes_conf_mat.metrics import MetricCollection, Metric, AveragedMetric
from bayes_conf_mat.experiment import ExperimentResult, SamplingMethod
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator
from bayes_conf_mat.experiment_aggregation.abc import ExperimentAggregationResult
from bayes_conf_mat.experiment_group import ExperimentGroup
from bayes_conf_mat.experiment_comparison import pairwise_compare, listwise_compare
from bayes_conf_mat.stats import summarize_posterior
from bayes_conf_mat.utils import (
    InMemoryCache,
    fmt,
    NotInCache,
)


class DistributionPlottingMethods(StrEnum):
    KDE = "kde"
    HIST = "hist"
    HISTOGRAM = "histogram"


class Study(Config):
    """This class represents a study, a collection of related experiments and experiment groups. It handles all lower level operations for you.

    You can use it to organize your experiments, compute and cache metrics, request analyses or figures, etc.

    Experiment groups should be directly comparable across groups.

    For example, a series of different models evaluated on the same dataset.

    Args:
        seed (int, optional): the random seed used to initialise the RNG. Defaults to the current time, in fractional seconds.
        num_samples (int, optional): the number of syntehtic confusion matrices to sample. A higher value is better, but more computationally expensive. Defaults to 10000, the minimum recommended value.
        ci_probability (float, optional): the size of the credibility intervals to compute. Defaults to 0.95, which is an arbitrary value, and should be carefully considered.
        experiments (dict[str, dict[str, dict[str, typing.Any]]], optional): a nested dict that contains (1) the experiment group name, (2) the experiment name, (3) and finally any IO/prior hyperparameters. Defaults to an empty dict.
        metrics (dict[str, dict[str, typing.Any]], optional): a nested dict that contains (1) the metric as metric syntax strings, (2) and any metric aggregation parameters. Defaults to an empty dict.
        cache_dir (str, optional): the location of the cache. Defaults to None.
        overwrite (bool, optional): whether to overwrite an existing cache. Defaults to False.
    """

    def __init__(
        self,
        seed: typing.Optional[int] = None,
        num_samples: typing.Optional[int] = None,
        ci_probability: typing.Optional[float] = None,
        experiments: dict[str, dict[str, dict[str, typing.Any]]] = {},
        metrics: dict[str, dict[str, typing.Any]] = {},
        # cache_dir: typing.Optional[str] = None,
        # overwrite: bool = False,
    ) -> None:
        # Instantiate the config back-end ======================================
        super().__init__(
            seed=seed,
            num_samples=num_samples,
            ci_probability=ci_probability,
            experiments=experiments,
            metrics=metrics,
        )

        # Instantiate the caching mechanism ====================================
        # self.cache_dir = cache_dir
        # self.overwrite = overwrite

        self.cache = InMemoryCache()

        # Instantiate the stores for experiments and metrics ===================
        # The experiment group store
        self._experiment_store = OrderedDict()

        # The collection of metrics
        self._metrics_store = MetricCollection()

        # The mapping from metric to aggregator
        self._metric_to_aggregator = dict()

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, typing.Any],
    ) -> typing.Self:
        """Creates a study from a dictionary.

        Keys and values should match pattern of output from Study.to_dict.

        Args:
            config_dict (dict[str, typing.Any]): the dictionary representation of the study configuration.
            cache_dir (str, optional): the location of the cache. Defaults to None.
            overwrite (bool, optional): whether to overwrite an existing cache. Defaults to False.

        Returns:
            typing.Self: an instance of a study
        """

        instance = super().from_dict(config_dict)

        # instance.cache_dir = cache_dir
        # instance.overwrite = overwrite

        return instance

    def _list_experiments(self) -> list[str]:
        """Returns a sorted list of all the experiments included in this self."""

        all_experiments = []
        for experiment_group, experiment_configs in self.experiments.items():
            for experiment_name, _ in experiment_configs.items():
                all_experiments.append(f"{experiment_group}/{experiment_name}")

        sorted(all_experiments)

        return all_experiments

    @cache
    def _compute_num_classes(self, fingerprint) -> int:
        """Returns the number of classes used in experiments in this study.

        Uses fingerprint to enable caching of result.

        Returns:
            int: the number of classes
        """

        all_num_classes = set()
        for experiment_group in self._experiment_store.values():
            all_num_classes.add(experiment_group.num_classes)

        if len(all_num_classes) > 1:
            raise ValueError(
                f"Inconsistent number of classes in experiment groups: {all_num_classes}"
            )

        return next(iter(all_num_classes))

    @property
    def num_classes(self) -> int:
        """Returns the number of classes used in experiments in this study.

        Returns:
            int: the number of classes
        """
        return self._compute_num_classes(fingerprint=self.fingerprint)

    def __repr__(self) -> str:
        return f"Study(experiments={self._list_experiments()}), metrics={self._metrics_store}"

    def __str__(self) -> str:
        return f"Study(experiments={self._list_experiments()}, metrics={self._metrics_store})"

    def __len__(self) -> int:
        return len(self._experiment_store)

    @staticmethod
    def _split_experiment_name(name: str, do_warn: bool = False) -> tuple[str, str]:
        """Tries to parse an experiment name string, e.g., "group/experiment"

        This enables the user to ignore experiment groups when not convenient.

        Args:
            name (str): the experiment name
            do_warn (bool, optional): whether to return warnings. Defaults to False.

        Returns:
            tuple[str, str]: the name of the experiment group and experiment
        """
        split_name = name.split("/")

        if len(split_name) == 2:
            experiment_group_name = split_name[0]
            experiment_name = split_name[1]

        elif len(split_name) == 1:
            experiment_group_name = split_name[0]
            experiment_name = split_name[0]

            if do_warn:
                warnings.warn(
                    f"Received experiment without experiment group: {experiment_name}. Adding to its own experiment group. To specify an experiment group, pass a string formatted as 'group/name'.",
                    category=ConfigWarning,
                )

        else:
            raise ConfigError(
                f"Received invalid experiment name. Currently: {name}. Must have at most 1 '/' character. Hierarchical experiment groups not (yet) implemented."
            )

        return experiment_group_name, experiment_name

    def _validate_experiment_name(self, name: str) -> str:
        experiment_group_name, experiment_name = self._split_experiment_name(
            name=name, do_warn=False
        )

        name = f"{experiment_group_name}/{experiment_name}"

        if experiment_name == "aggregated":
            if experiment_group_name not in self._experiment_store:
                raise ValueError(
                    f"Experiment group {experiment_group_name} does not (yet) exist."
                )

        elif name not in self._list_experiments():
            raise ValueError(f"Experiment {name} does not (yet) exist.")

        return name

    def _validate_metric_class_label_combination(
        self,
        metric: str | MetricLike,
        class_label: typing.Optional[int],
    ) -> tuple[MetricLike, int]:
        try:
            metric_: MetricLike = self._metrics_store[metric]
        except KeyError:
            raise KeyError(
                f"Could not find metric '{metric}' in the metrics collection. Consider adding it using `self.add_metric`"
            )

        if metric_.is_multiclass:
            if not ((class_label == 0) or (class_label is None)):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0
        else:
            if class_label is None:
                raise ValueError(
                    f"Metric '{metric_.name}' is not multiclass. You must provide a class label."
                )
            elif class_label < 0 or class_label > self.num_classes - 1:
                raise ValueError(
                    f"Study only has {self.num_classes} classes. Class label must be in range [0, {self.num_classes - 1}]. Currently {class_label}."
                )

        return metric_, class_label

    def add_experiment(
        self,
        experiment_name: str,
        confusion_matrix: typing.Optional[
            jtyping.Int[np.typing.ArrayLike, " num_classes num_classes"]
        ] = None,
        prevalence_prior: typing.Optional[
            str | float | jtyping.Float[np.typing.ArrayLike, " num_classes"]
        ] = None,
        confusion_prior: typing.Optional[
            str | float | jtyping.Float[np.typing.ArrayLike, " num_classes num_classes"]
        ] = None,
        **io_kwargs,
    ) -> None:
        """Adds an experiment to this study.

        Args:
            experiment_name (str): the name of the experiment and experiment group. Should be written as 'experiment_group/experiment'. If the experiment group name is omitted, the experiment gets added to a new experiment group of the same name.
            confusion_matrix (Optional[Int[ArrayLike, 'num_classes num_classes']]): the confusion matrix for this experiment. If left as `None`, then the IO kwargs should specify where a confusion matrix might be found
            prevalence_prior (Optional[str | float | Float[ArrayLike, ' num_classes'] ], optional): the prior over the prevalence counts for this experiments. Defaults to 0, Haldane's prior.
            confusion_prior (Optional[str | float | Float[ArrayLike, ' num_classes num_classes'] ], optional): the prior over the confusion counts for this experiments. Defaults to 0, Haldane's prior.

        Example:
            Add an experiment named 'test_a' to experiment group 'test'
            >>> self.add_experiment(
                name="test/test_a",
                confusion_matrix=[[1, 0], [0, 1]],
            )

        """

        # Parse experiment group and experiment name ===========================
        experiment_group_name, experiment_name = self._split_experiment_name(
            experiment_name
        )

        # Type checking ========================================================
        # If passing a list o np.ndarray as the confusion matrix, wraps it into
        # a dict to be fed to an IO method
        conf_mat_io_config: dict[str, typing.Any] = dict(io_kwargs)

        if isinstance(confusion_matrix, (list, tuple, np.ndarray)):
            conf_mat_io_config.update(dict(confusion_matrix=confusion_matrix))

        # Create a complete IO config for the confusion matrix
        conf_mat_io_config.update(
            {
                "prevalence_prior": prevalence_prior,
                "confusion_prior": confusion_prior,
            }
        )

        # Add the experiment to the config back-end ============================
        cur_experiments = self.experiments
        if experiment_group_name not in cur_experiments:
            cur_experiments[experiment_group_name] = dict()

        cur_experiments[experiment_group_name].update(
            {experiment_name: conf_mat_io_config}
        )

        self.experiments = cur_experiments

        # Add the experiment and experiment_group to the store =================
        # Get the experiment group if it exists, otherwise create it
        if experiment_group_name not in self._experiment_store:
            # Give the new experiment group its own RNG
            # Should be independent from the self's RNG and all other
            # experimentgroups' RNGs
            indep_rng = self.rng.spawn(n_children=1)[0]

            experiment_group = ExperimentGroup(
                name=experiment_group_name,
                rng=indep_rng,
            )

            self._experiment_store[experiment_group_name] = experiment_group

        experiment_config = self.experiments[experiment_group_name][experiment_name]

        # Finally, add the experiment to the right experiment group
        experiment_group = self._experiment_store[experiment_group_name].add_experiment(
            name=experiment_name,
            confusion_matrix=experiment_config["confusion_matrix"],
            prevalence_prior=experiment_config["prevalence_prior"],
            confusion_prior=experiment_config["confusion_prior"],
        )

    def add_metric(
        self,
        metric: str | MetricLike,
        aggregation: typing.Optional[str] = None,
        **aggregation_kwargs,
    ) -> None:
        """Adds a metric to the self. If the is more than one experiment in an experiment group, add an aggregation method.

        Args:
            metric (str | MetricLike): the metric to be added
            aggregation (str, optional): the name of the aggregation method. Defaults to None.
            aggregation_kwargs: keyword arguments passed to the `get_experiment_aggregator` function
        """

        # Try to figure out the metric name
        if isinstance(metric, Metric) or isinstance(metric, AveragedMetric):
            metric_name: str = metric.name
        else:
            metric_name: str = metric  # type: ignore

        # Retrieve the current set of metrics
        cur_metrics = self.metrics
        if aggregation is None:
            cur_metrics[metric_name] = dict()
        else:
            cur_metrics[metric_name] = (
                dict(aggregation=aggregation) | aggregation_kwargs
            )

        # Update the stored set of metrics
        # Applies validation
        self.metrics = cur_metrics

        # Add the metric to the self ==========================================
        self._metrics_store.add(metric=metric)

        # Add a cross-experiment aggregator to the metric =====================
        if len(self.metrics[metric_name]) != 0:
            indep_rng = self.rng.spawn(n_children=1)[0]

            aggregator = get_experiment_aggregator(
                rng=indep_rng, **self.metrics[metric_name]
            )
            self._metric_to_aggregator[self._metrics_store[metric]] = aggregator

    def _sample_metrics(self, sampling_method: str):
        match sampling_method:
            case (
                SamplingMethod.POSTERIOR.value
                | SamplingMethod.PRIOR.value
                | SamplingMethod.RANDOM.value
            ):
                # Compute metrics for the entire experiment group
                for experiment_group in self._experiment_store.values():
                    experiment_group_results = experiment_group.sample_metrics(
                        metrics=self._metrics_store,
                        sampling_method=sampling_method,
                        num_samples=self.num_samples,
                        metric_to_aggregator=self._metric_to_aggregator,
                    )

                    # Cache the aggregated results
                    for (
                        metric,
                        aggregation_result,
                    ) in experiment_group_results.aggregation_result.items():
                        self.cache.cache(
                            fingerprint=self.fingerprint,
                            keys=[
                                metric.name,
                                experiment_group.name,
                                "aggregated",
                                sampling_method,
                            ],
                            value=aggregation_result,
                        )

                    # Cache the individual experiment results
                    for (
                        metric,
                        individual_experiment_results,
                    ) in experiment_group_results.individual_experiment_results.items():
                        for (
                            experiment,
                            experiment_result,
                        ) in individual_experiment_results.items():
                            self.cache.cache(
                                fingerprint=self.fingerprint,
                                keys=[
                                    metric.name,
                                    experiment_group.name,
                                    experiment.name,
                                    sampling_method,
                                ],
                                value=experiment_result,
                            )

            case SamplingMethod.INPUT.value:
                # Compute metrics for the entire experiment group
                for experiment_group in self._experiment_store.values():
                    for experiment in experiment_group.experiments.values():
                        experiment_results = experiment.sample_metrics(
                            metrics=self._metrics_store,
                            sampling_method="input",
                            num_samples=self.num_samples,
                        )

                        for metric, experiment_result in experiment_results.items():
                            self.cache.cache(
                                fingerprint=self.fingerprint,
                                keys=[
                                    metric.name,
                                    experiment_group.name,
                                    experiment.name,
                                    "input",
                                ],
                                value=experiment_result,
                            )

            case _:
                raise ValueError(
                    f"Parameter `sampling_method` must be one of {tuple(sm.value for sm in SamplingMethod)}. Currently: {sampling_method}"
                )

    def get_metric_samples(
        self,
        metric: str | MetricLike,
        experiment_name: str,
        sampling_method: str,
    ) -> typing.Union[ExperimentResult, ExperimentAggregationResult]:
        """Loads or computes samples for a metric, belonging to an experiment.

        Args:
            metric (str | MetricLike): the name of the metric
            experiment_name (str): the name of the experiment. You can also pass 'experiment_group/aggregated' to retrieve the aggregated metric values.
            sampling_method (str): the sampling method used to generate the metric values. Must a member of the SamplingMethod enum

        Returns:
            typing.Union[ExperimentResult, ExperimentAggregationResult]

        Example:
            >>> experiment_result = self.get_metric_samples(metric="accuracy", sampling_method="posterior", experiment_name="test/test_a")
            ExperimentResult(experiment=ExperimentGroup(test_a), metric=Metric(accuracy))

            >>> experiment_result = self.get_metric_samples(metric="accuracy", sampling_method="posterior", experiment_name="test/aggregated")
            ExperimentAggregationResult(experiment_group=ExperimentGroup(test), metric=Metric(accuracy), aggregator=ExperimentAggregator(fe_gaussian))

        """

        if isinstance(metric, Metric | AveragedMetric):
            metric = metric.name

        # Validate the experiment name before trying to fetch its values
        experiment_name = self._validate_experiment_name(experiment_name)
        experiment_group_name, _experiment_name = self._split_experiment_name(
            experiment_name
        )

        keys = [metric, experiment_group_name, _experiment_name, sampling_method]

        if self.cache.isin(fingerprint=self.fingerprint, keys=keys):
            result: ExperimentResult | ExperimentAggregationResult | NotInCache = (  # type: ignore
                self.cache.load(fingerprint=self.fingerprint, keys=keys)
            )

        else:
            self._sample_metrics(sampling_method=sampling_method)

            result: ExperimentResult | ExperimentAggregationResult | NotInCache = (  # type: ignore
                self.cache.load(fingerprint=self.fingerprint, keys=keys)
            )

        if result is NotInCache:
            raise ValueError(
                f"Got a NotInCache for {keys}. Cannot continue. Please report this issue."
            )
        else:
            result: ExperimentResult | ExperimentAggregationResult

        return result

    def _construct_metric_summary_table(
        self,
        metric: MetricLike,
        sampling_method: str,
        class_label: typing.Optional[int] = None,
        table_fmt: str = "html",
        precision: int = 4,
        include_observed_values: bool = False,
    ) -> str:
        import tabulate

        table = []
        for experiment_group_name, experiment_group in self._experiment_store.items():
            for experiment_name, _ in experiment_group.experiments.items():
                sampled_experiment_result = self.get_metric_samples(
                    metric=metric.name,
                    experiment_name=f"{experiment_group_name}/{experiment_name}",
                    sampling_method=sampling_method,
                )

                distribution_summary = summarize_posterior(
                    sampled_experiment_result.values[:, class_label],
                    ci_probability=self.ci_probability,  # type: ignore
                )

                if distribution_summary.hdi[1] - distribution_summary.hdi[0] > 1e-4:
                    hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='f')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='f')}]"
                else:
                    hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='e')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='e')}]"

                table_row = [
                    experiment_group_name,
                    experiment_name,
                ]

                if include_observed_values:
                    observed_experiment_result = self.get_metric_samples(
                        metric=metric.name,
                        experiment_name=f"{experiment_group_name}/{experiment_name}",
                        sampling_method=SamplingMethod.INPUT,
                    )

                    table_row.append(observed_experiment_result.values[:, class_label])

                table_row += [
                    distribution_summary.median,
                    distribution_summary.mode,
                    hdi_str,
                    distribution_summary.metric_uncertainty,
                    distribution_summary.skew,
                    distribution_summary.kurtosis,
                ]

                table.append(table_row)

        headers = ["Group", "Experiment"]
        if include_observed_values:
            headers += ["Observed"]

        headers += [*distribution_summary.headers]  # type: ignore

        table = tabulate.tabulate(  # type: ignore
            tabular_data=table,
            headers=headers,
            floatfmt=f".{precision}f",
            colalign=["left", "left"] + ["decimal" for _ in headers[2:]],
            tablefmt=table_fmt,
        )

        return table

    def report_metric_summaries(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        table_fmt: str = "html",
        precision: int = 4,
    ) -> str:
        """Generates a table with summary statistics for all experiments

        Args:
            metric (str): the name of the metric
            class_label (typing.Optional[int], optional): the class label. Leave 0 or None if using a multiclass metric. Defaults to None.
            table_fmt (str, optional): the format of the table, passed to [tabulate](https://github.com/astanin/python-tabulate#table-format). Defaults to "html".
            precision (int, optional): the required precision of the presented numbers. Defaults to 4.

        Returns:
            str: the table as a string
        """
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        table = self._construct_metric_summary_table(
            metric=metric,
            class_label=class_label,
            sampling_method=SamplingMethod.POSTERIOR,
            table_fmt=table_fmt,
            precision=precision,
            include_observed_values=True,
        )

        return table

    def report_random_metric_summaries(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        table_fmt: str = "html",
        precision: int = 4,
    ) -> str:
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        table = self._construct_metric_summary_table(
            metric=metric,
            class_label=class_label,
            sampling_method="random",
            table_fmt=table_fmt,
            precision=precision,
            include_observed_values=False,
        )

        return table

    def plot_metric_summaries(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        method: str = "kde",
        bandwidth: float = 1.0,
        bins: int | list[int] | str = "auto",
        normalize: bool = False,
        figsize: typing.Optional[tuple[float, float]] = None,
        fontsize: float = 9.0,
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
        obs_point_size: float | None = None,
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
    ) -> matplotlib.figure.Figure:
        """Plots the distrbution of sampled metric values for a particular metric and class combination.

        Args:
            metric (str): the name of the metric
            class_label (int | None, optional): the class label. Defaults to None.
            observed_values (dict[str, ExperimentResult]): the observed metric values
            sampled_values (dict[str, ExperimentResult]): the sampled metric values
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
            plot_base_line (bool, optional): whether to plot a line at the base of the distribution. Defaults to True.
            base_lines_colour (str, optional): the colour of the base line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
            base_lines_format (str, optional): the format of the base line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
            plot_experiment_name (bool, optional): whether to plot the experiment names as labels. Defaults to True.

        Returns:
            matplotlib.figure.Figure: the completed figure of the distribution plot
        """
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        # Import optional dependencies
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
            )

        total_num_experiments = len(self._list_experiments())

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
        for experiment_group_name, experiment_group in self._experiment_store.items():
            for experiment_name, _ in experiment_group.experiments.items():
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

                distribution_samples = self.get_metric_samples(
                    metric=metric.name,
                    experiment_name=f"{experiment_group_name}/{experiment_name}",
                    sampling_method=SamplingMethod.POSTERIOR,
                ).values[:, class_label]

                # Get summary statistics
                posterior_summary = summarize_posterior(
                    posterior_samples=distribution_samples,
                    ci_probability=self.ci_probability,  # type: ignore
                )

                all_medians.append(posterior_summary.median)
                all_hdi_ranges.append(
                    posterior_summary.hdi[1] - posterior_summary.hdi[0]
                )

                match method:
                    case DistributionPlottingMethods.KDE.value:
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
                                kde_x,
                                kde_y,
                                color=area_colour,
                                zorder=0,
                                alpha=area_alpha,
                            )

                    case (
                        DistributionPlottingMethods.HIST.value
                        | DistributionPlottingMethods.HISTOGRAM.value
                    ):
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

                    case _:
                        del fig, axes
                        raise ValueError(
                            f"Parameter `method` must be one of {tuple(sm.value for sm in DistributionPlottingMethods)}. Currently: {method}"
                        )

                if plot_obs_point:
                    # Add a point for the true point value
                    observed_metric_value = self.get_metric_samples(
                        metric=metric.name,
                        experiment_name=f"{experiment_group_name}/{experiment_name}",
                        sampling_method=SamplingMethod.INPUT,
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
                        xp=kde_x,
                        fp=kde_y,
                    )

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
                        xp=kde_x,
                        fp=kde_y,
                    )

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
                        xp=kde_x,
                        fp=kde_y,
                    )

                    axes[i].vlines(
                        x_hdi_ub,
                        0,
                        y_hdi_ub,
                        color=hdi_lines_colour,
                        linestyle=hdi_line_format,
                        zorder=1,
                    )

                i += 1

        smallest_hdi_range = np.min(all_hdi_ranges)

        # Clip the gran max and min to avoid huge positive or negative outliers
        # resulting in tiny distributions
        grand_min_x = max(
            np.min(all_min_x),
            np.min(all_medians) - 5 * smallest_hdi_range,
        )
        grand_max_x = min(
            np.max(all_max_x),
            np.max(all_medians) + 5 * smallest_hdi_range,
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

    def _pairwise_compare(
        self,
        metric: MetricLike,
        class_label: int,
        experiment_a: str,
        experiment_b: str,
        min_sig_diff: typing.Optional[float] = None,
    ) -> PairwiseComparisonResult:
        lhs_samples = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment_a,
            sampling_method=SamplingMethod.POSTERIOR,
        ).values[:, class_label]

        rhs_samples = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment_b,
            sampling_method=SamplingMethod.POSTERIOR,
        ).values[:, class_label]

        lhs_random_samples = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment_a,
            sampling_method=SamplingMethod.RANDOM,
        ).values[:, class_label]

        rhs_random_samples = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment_b,
            sampling_method=SamplingMethod.RANDOM,
        ).values[:, class_label]

        if "aggregated" not in experiment_a and "aggregated" not in experiment_b:
            lhs_observed = self.get_metric_samples(
                metric=metric.name,
                experiment_name=experiment_a,
                sampling_method=SamplingMethod.INPUT,
            ).values[:, class_label]

            rhs_observed = self.get_metric_samples(
                metric=metric.name,
                experiment_name=experiment_b,
                sampling_method=SamplingMethod.INPUT,
            ).values[:, class_label]

            observed_diff = lhs_observed - rhs_observed
        else:
            observed_diff = None

        comparison_result = pairwise_compare(
            metric=metric,
            diff_dist=lhs_samples - rhs_samples,
            random_diff_dist=lhs_random_samples - rhs_random_samples,
            ci_probability=self.ci_probability,  # type: ignore
            min_sig_diff=min_sig_diff,
            observed_difference=observed_diff,
            lhs_name=experiment_a,
            rhs_name=experiment_b,
        )

        return comparison_result

    def report_pairwise_comparison(
        self,
        metric: str,  # type: ignore
        experiment_a: str,
        experiment_b: str,
        class_label: typing.Optional[int] = None,  # type: ignore
        min_sig_diff: typing.Optional[float] = None,
        precision: int = 4,
    ) -> str:
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        comparison_result = self._pairwise_compare(
            metric=metric,
            class_label=class_label,
            experiment_a=experiment_a,
            experiment_b=experiment_b,
            min_sig_diff=min_sig_diff,
        )

        return comparison_result.template_sentence(precision=precision)

    def report_pairwise_comparison_plot(
        self,
        metric: str,  # type: ignore
        experiment_a: str,
        experiment_b: str,
        class_label: int = None,  # type: ignore
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
    ) -> matplotlib.figure.Figure:
        try:
            # Import optional dependencies
            import matplotlib.pyplot as plt
            import seaborn as sns

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
            )

        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        comparison_result = self._pairwise_compare(
            metric=metric,
            class_label=class_label,
            experiment_a=experiment_a,
            experiment_b=experiment_b,
            min_sig_diff=min_sig_diff,
        )

        # Figure out the maximum and minimum of the difference distribution
        diff_bounds = (
            metric.bounds[0] - metric.bounds[1],
            metric.bounds[1] - metric.bounds[0],
        )

        # Figure instantiation
        if figsize is None:
            # Try to set a decent default figure size
            _figsize = (6.30, 2.52)
        else:
            _figsize = figsize

        fig, ax = plt.subplots(1, 1, figsize=_figsize)

        match method:
            case DistributionPlottingMethods.KDE.value:
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

                kde_x: np.ndarray = kdeline.get_xdata()  # type: ignore
                kde_y: np.ndarray = kdeline.get_ydata()  # type: ignore

            case (
                DistributionPlottingMethods.HIST.value
                | DistributionPlottingMethods.HISTOGRAM.value
            ):
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

                kde_x: np.ndarray = kdeline.get_xdata()  # type: ignore
                kde_y: np.ndarray = kdeline.get_ydata()  # type: ignore

                kde_x = np.repeat(kde_x, 2)
                kde_y = np.concatenate([[0], np.repeat(kde_y, 2)[:-1]])

            case _:
                del fig, ax
                raise ValueError(
                    f"Parameter `method` must be one of {tuple(sm.value for sm in DistributionPlottingMethods)}. Currently: {method}"
                )

        # Compute the actual maximum and minimum of the difference distribution
        min_x = np.min(kde_x)
        max_x = np.max(kde_x)

        if plot_min_sig_diff_lines:
            for msd in [
                -comparison_result.min_sig_diff,
                comparison_result.min_sig_diff,
            ]:
                y_msd = np.interp(
                    x=msd,
                    xp=kde_x,
                    fp=kde_y,
                )

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
            num=2 * kde_x.shape[0],
        )

        rope_yy = np.interp(
            x=rope_xx,
            xp=kde_x,
            fp=kde_y,
        )

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
            min_x, -comparison_result.min_sig_diff, num=2 * kde_x.shape[0]
        )

        neg_sig_yy = np.interp(
            x=neg_sig_xx,
            xp=kde_x,
            fp=kde_y,
        )

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
            comparison_result.min_sig_diff, max_x, num=2 * kde_x.shape[0]
        )

        pos_sig_yy = np.interp(
            x=pos_sig_xx,
            xp=kde_x,
            fp=kde_y,
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
                xp=kde_x,
                fp=kde_y,
            )

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
                    xp=kde_x,
                    fp=kde_y,
                )

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
                    xp=kde_x,
                    fp=kde_y,
                )

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
                s=f"$p_{{ROPE}}$\n{fmt(comparison_result.p_rope, precision=precision, mode='%')}\n",
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

    def _pairwise_random_comparison(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int],  # type: ignore
        experiment: str,
        min_sig_diff: typing.Optional[float] = None,
    ) -> PairwiseComparisonResult:
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        actual_result = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment,
            sampling_method=SamplingMethod.POSTERIOR,
        ).values[:, class_label]

        random_results = self.get_metric_samples(
            metric=metric.name,
            experiment_name=experiment,
            sampling_method=SamplingMethod.RANDOM,
        ).values[:, class_label]

        comparison_result = pairwise_compare(
            metric=metric,
            diff_dist=actual_result - random_results,
            random_diff_dist=None,
            ci_probability=self.ci_probability,  # type: ignore
            min_sig_diff=min_sig_diff,
            observed_difference=None,
            lhs_name=experiment,
            rhs_name="random",
        )

        return comparison_result

    def report_pairwise_comparison_to_random(
        self,
        metric: str,
        class_label: int | None = None,
        table_fmt: str = "html",
        precision: int = 4,
        min_sig_diff: typing.Optional[float] = None,
    ) -> str:
        import tabulate

        records = []

        for experiment in self._list_experiments():
            random_comparison_result = self._pairwise_random_comparison(
                metric=metric,
                class_label=class_label,
                experiment=experiment,
                min_sig_diff=min_sig_diff,
            )

            experiment_group_name, experiment_name = self._split_experiment_name(
                experiment
            )

            rope_lb = fmt(
                -random_comparison_result.min_sig_diff, precision=precision, mode="f"
            )
            rope_ub = fmt(
                random_comparison_result.min_sig_diff, precision=precision, mode="f"
            )

            random_comparison_record = {
                "Group": experiment_group_name,
                "Experiment": experiment_name,
                "Median Δ": random_comparison_result.diff_dist_summary.median,
                "p_direction": random_comparison_result.p_direction,
                "ROPE": f"[{rope_lb}, {rope_ub}]",
                "p_ROPE": random_comparison_result.p_rope,
                "p_sig": random_comparison_result.p_bi_sig,
            }

            records.append(random_comparison_record)

        table = tabulate.tabulate(  # type: ignore
            tabular_data=records,
            headers="keys",
            floatfmt=f".{precision}f",
            colalign=["left", "left"]
            + ["decimal" for _ in range(len(records[0].keys()) - 2)],
            tablefmt=table_fmt,
        )

        return table

    def report_listwise_comparison(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        table_fmt: str = "html",
        precision: int = 4,
    ):
        """Reports the probability for an experiment to achieve a rank when compared to all other experiments on the same metric.

        Args:
            metric (str): the name of the metric
            class_label (int | None, optional): the class label. Leave 0 or None if using a multiclass metric. Defaults to None.
            table_fmt (str, optional): the format of the table, passed to [tabulate](https://github.com/astanin/python-tabulate#table-format). Defaults to "html".
            precision (int, optional): the required precision of the presented numbers. Defaults to 4.

        Returns:
            str: the table as a string
        """
        import tabulate

        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        # TODO: should this comparison happen for all experiments
        # or for each experiment group?
        experiment_values = {
            experiment: self.get_metric_samples(
                experiment_name=experiment,
                metric=metric.name,
                sampling_method=SamplingMethod.POSTERIOR,
            ).values[:, class_label]
            for experiment in self._list_experiments()
        }

        p_experiment_given_rank_arr = listwise_compare(
            experiment_values_dict=experiment_values,
            metric_name=metric.name,
        ).p_experiment_given_rank

        headers = ["Group", "Experiment"] + [
            f"Rank {i + 1}" for i in range(p_experiment_given_rank_arr.shape[0])
        ]

        table = tabulate.tabulate(  # type: ignore
            tabular_data=[
                [*self._split_experiment_name(experiment_names), *row]
                for row, experiment_names in zip(
                    p_experiment_given_rank_arr, self._list_experiments()
                )
            ],
            tablefmt=table_fmt,
            floatfmt=f".{precision}f",
            headers=headers,
            colalign=["left", "left"] + ["decimal" for _ in headers[2:]],
        )

        return table

    def report_aggregated_metric_summaries(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        precision: int = 4,
        table_fmt: str = "html",
    ) -> str:
        import tabulate

        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        table = []
        for experiment_group in list(self.experiments.keys()):
            experiment_aggregation_result: ExperimentAggregationResult = (
                self.get_metric_samples(
                    metric=metric.name,
                    experiment_name=f"{experiment_group}/aggregated",
                    sampling_method=SamplingMethod.POSTERIOR,
                )
            )  # type: ignore

            distribution_summary = summarize_posterior(
                posterior_samples=experiment_aggregation_result.values[:, class_label],
                ci_probability=self.ci_probability,  # type: ignore
            )

            if distribution_summary.hdi[1] - distribution_summary.hdi[0] > 1e-4:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='f')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='f')}]"
            else:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='e')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='e')}]"

            heterogeneity_result = experiment_aggregation_result.heterogeneity_results[
                class_label
            ]

            table_row = [
                experiment_group,
                distribution_summary.median,
                distribution_summary.mode,
                hdi_str,
                distribution_summary.metric_uncertainty,
                distribution_summary.skew,
                distribution_summary.kurtosis,
                fmt(
                    heterogeneity_result.within_experiment_variance,
                    precision=precision,
                    mode="f",
                ),
                fmt(
                    heterogeneity_result.between_experiment_variance,
                    precision=precision,
                    mode="f",
                ),
                fmt(heterogeneity_result.i2, precision=precision, mode="%"),
            ]

            table.append(table_row)

        headers = [
            "Group",
            *distribution_summary.headers,  # type: ignore
            "Var. Within",
            "Var. Between",
            "I2",
        ]

        table = tabulate.tabulate(  # type: ignore
            tabular_data=table,
            headers=headers,
            floatfmt=f".{precision}f",
            colalign=["left"] + ["decimal" for _ in headers[1:]],
            tablefmt=table_fmt,
        )

        return table

    def plot_experiment_aggregation(
        self,
        metric: str,  # type: ignore
        experiment_group: str,
        class_label: typing.Optional[int] = None,  # type: ignore
        method: str = "kde",
        bandwidth: float = 1.0,
        bins: int | list[int] | str = "auto",
        normalize: bool = False,
        figsize: typing.Optional[tuple[float, float]] = None,
        fontsize: float = 9.0,
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
        extrema_line_height: float = 12.0,
        extrema_lines_width: float = 1.0,
        plot_base_line: bool = True,
        base_line_colour: str = "black",
        base_line_format: str = "-",
        base_line_width: int = 1,
        plot_experiment_name: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plots the distrbution of sampled metric values for a specific experiment group, with the aggregated distribution, for a particular metric and class combination.

        Args:
            metric (str): the name of the metric
            experiment_group (str): the name of the experiment group
            class_label (int | None, optional): the class label. Defaults to None.
            observed_values (dict[str, ExperimentResult]): the observed metric values
            sampled_values (dict[str, ExperimentResult]): the sampled metric values
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
            plot_base_line (bool, optional): whether to plot a line at the base of the distribution. Defaults to True.
            base_lines_colour (str, optional): the colour of the base line. Corresponds to [matplotlib's `color` parameter](https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def). Defaults to "black".
            base_lines_format (str, optional): the format of the base line. Corresponds to [matplotlib's `linestyle` parameter](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html). Defaults to "-".
            plot_experiment_name (bool, optional): whether to plot the experiment names as labels. Defaults to True.

        Returns:
            matplotlib.figure.Figure: the completed figure of the distribution plot
        """

        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        # Import optional dependencies
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
            )

        total_num_experiments = len(self._experiment_store[experiment_group]) + 1

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
        all_hdi_ranges = []
        for experiment_name, _ in self._experiment_store[
            experiment_group
        ].experiments.items():
            if plot_experiment_name:
                # Set the axis title
                # Needs to happen before KDE
                axes[i].set_ylabel(
                    experiment_name,
                    rotation=0,
                    ha="right",
                    fontsize=fontsize,
                )

            distribution_samples = self.get_metric_samples(
                metric=metric.name,
                experiment_name=f"{experiment_group}/{experiment_name}",
                sampling_method=SamplingMethod.POSTERIOR,
            ).values[:, class_label]

            # Get summary statistics
            posterior_summary = summarize_posterior(
                distribution_samples,
                ci_probability=self.ci_probability,  # type: ignore
            )

            all_hdi_ranges.append(posterior_summary.hdi[1] - posterior_summary.hdi[0])

            match method:
                case DistributionPlottingMethods.KDE.value:
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
                            kde_x,
                            kde_y,
                            color=area_colour,
                            zorder=0,
                            alpha=area_alpha,
                        )

                case (
                    DistributionPlottingMethods.HIST.value
                    | DistributionPlottingMethods.HISTOGRAM.value
                ):
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

                case _:
                    del fig, axes
                    raise ValueError(
                        f"Parameter `method` must be one of {tuple(sm.value for sm in DistributionPlottingMethods)}. Currently: {method}"
                    )

            if plot_obs_point:
                # Add a point for the true point value
                observed_metric_value = self.get_metric_samples(
                    metric=metric.name,
                    experiment_name=f"{experiment_group}/{experiment_name}",
                    sampling_method=SamplingMethod.INPUT,
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
                    xp=kde_x,
                    fp=kde_y,
                )

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
                    xp=kde_x,
                    fp=kde_y,
                )

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
                    xp=kde_x,
                    fp=kde_y,
                )

                axes[i].vlines(
                    x_hdi_ub,
                    0,
                    y_hdi_ub,
                    color=hdi_lines_colour,
                    linestyle=hdi_line_format,
                    zorder=1,
                )

            i += 1

        # ==============================================================================
        # Plot the aggregated distribution
        # ==============================================================================
        if plot_experiment_name:
            # Set the axis title
            # Needs to happen before KDE
            axes[-1].set_ylabel(
                "Aggregated",
                rotation=0,
                va="center",
                ha="right",
                fontsize=fontsize,
            )

        agg_distribution_samples = self.get_metric_samples(
            metric=metric.name,
            experiment_name=f"{experiment_group}/aggregated",
            sampling_method=SamplingMethod.POSTERIOR,
        ).values[:, class_label]

        # Get summary statistics
        aggregated_summary = summarize_posterior(
            agg_distribution_samples,
            ci_probability=self.ci_probability,  # type: ignore
        )

        match method:
            case DistributionPlottingMethods.KDE.value:
                # Plot the kde
                sns.kdeplot(
                    agg_distribution_samples,
                    fill=False,
                    bw_adjust=bandwidth,
                    ax=axes[-1],
                    color=edge_colour,
                    clip=metric_bounds,
                    zorder=2,
                )

                kdeline = axes[-1].lines[0]

                kde_x = kdeline.get_xdata()
                kde_y = kdeline.get_ydata()

                all_min_x.append(kde_x[0])
                all_max_x.append(kde_x[-1])
                all_max_height.append(np.max(kde_y))

                if area_colour is not None:
                    axes[-1].fill_between(
                        kde_x,
                        kde_y,
                        color=area_colour,
                        zorder=0,
                        alpha=area_alpha,
                    )

            case (
                DistributionPlottingMethods.HIST.value
                | DistributionPlottingMethods.HISTOGRAM.value
            ):
                sns.histplot(
                    agg_distribution_samples,
                    fill=False,
                    bins=bins,
                    stat="density",
                    element="step",
                    ax=axes[-1],
                    color=edge_colour,
                    zorder=2,
                )

                kdeline = axes[-1].lines[0]

                kde_x = kdeline.get_xdata()
                kde_y = kdeline.get_ydata()

                all_min_x.append(kde_x[0])
                all_max_x.append(kde_x[-1])
                all_max_height.append(np.max(kde_y))

                kde_x = np.repeat(kde_x, 2)
                kde_y = np.concatenate([[0], np.repeat(kde_y, 2)[:-1]])

                if area_colour is not None:
                    axes[-1].fill_between(
                        kde_x,
                        kde_y,
                        color=area_colour,
                        zorder=0,
                        alpha=area_alpha,
                    )

            case _:
                del fig, axes
                raise ValueError(
                    f"Parameter `method` must be one of {tuple(sm.value for sm in DistributionPlottingMethods)}. Currently: {method}"
                )

        if plot_median_line:
            # Plot median line
            median_x = aggregated_summary.median

            y_median = np.interp(
                x=median_x,
                xp=kde_x,
                fp=kde_y,
            )

            axes[-1].vlines(
                median_x,
                0,
                y_median,
                color=median_line_colour,
                linestyle=median_line_format,
                zorder=1,
            )

        if plot_hdi_lines:
            x_hdi_lb = aggregated_summary.hdi[0]

            y_hdi_lb = np.interp(
                x=x_hdi_lb,
                xp=kde_x,
                fp=kde_y,
            )

            axes[-1].vlines(
                x_hdi_lb,
                0,
                y_hdi_lb,
                color=hdi_lines_colour,
                linestyle=hdi_line_format,
                zorder=1,
            )

            x_hdi_ub = aggregated_summary.hdi[1]

            y_hdi_ub = np.interp(
                x=x_hdi_ub,
                xp=kde_x,
                fp=kde_y,
            )

            axes[-1].vlines(
                x_hdi_ub,
                0,
                y_hdi_ub,
                color=hdi_lines_colour,
                linestyle=hdi_line_format,
                zorder=1,
            )

        # ==============================================================================
        # Determine the plotting domain
        # ==============================================================================
        smallest_hdi_range = np.min(all_hdi_ranges)

        # Clip the grand max and min to avoid huge positive or negative outliers
        # resulting in tiny distributions
        grand_min_x = max(
            metric_bounds[0],
            # np.min(all_min_x),
            np.min(aggregated_summary.median) - 5 * smallest_hdi_range,
        )
        grand_max_x = min(
            metric_bounds[1],
            # np.max(all_max_x),
            np.max(aggregated_summary.median) + 5 * smallest_hdi_range,
        )

        cur_xlim_min = aggregated_summary.median - np.abs(
            grand_min_x - aggregated_summary.median
        )
        cur_xlim_max = aggregated_summary.median + np.abs(
            grand_max_x - aggregated_summary.median
        )

        for ax in axes:
            ax.set_xlim(
                cur_xlim_min,
                cur_xlim_max,
            )

        # ==============================================================================
        # Base line, extrema lines, spine
        # ==============================================================================
        for i, ax in enumerate(axes):
            if plot_base_line:
                # Add base line
                ax.hlines(
                    0,
                    max(all_min_x[i], cur_xlim_max),  # type: ignore
                    min(all_max_x[i], cur_xlim_min),  # type: ignore
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
                if all_min_x[i] >= cur_xlim_min:
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

                if all_max_x[i] <= cur_xlim_max:
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
        axes[-1].tick_params(
            axis="x", labelsize=axis_fontsize if axis_fontsize is not None else fontsize
        )

        axes[-1].set_xlabel(metric.full_name, fontsize=fontsize)

        fig.suptitle(
            f"Group {experiment_group} w/ {self._metric_to_aggregator[metric].full_name}",
            fontsize=fontsize,
        )

        fig.tight_layout()

        return fig

    # TODO: document this method
    def plot_forest_plot(
        self,
        metric: str,  # type: ignore
        class_label: typing.Optional[int] = None,  # type: ignore
        figsize: typing.Optional[tuple[float, float]] = None,
        fontsize: float = 9.0,
        axis_fontsize: typing.Optional[float] = None,
        fontname: str = "monospace",
        median_marker: str = "s",
        median_marker_edge_colour: str = "black",
        median_marker_face_colour: str = "white",
        median_marker_size: float = 7,
        median_marker_line_width: float = 1.5,
        agg_offset: int = 1,
        agg_median_marker: str = "D",
        agg_median_marker_edge_colour: str = "black",
        agg_median_marker_face_colour: str = "white",
        agg_median_marker_size: float = 10,
        agg_median_marker_line_width: float = 1.5,
        hdi_lines_colour: str = "black",
        hdi_lines_format: str = "-",
        hdi_lines_width: int = 1,
        plot_agg_median_line: bool = True,
        agg_median_line_colour: str = "black",
        agg_median_line_format: str = "--",
        agg_median_line_width: float = 1.0,
        plot_experiment_name: bool = True,
        experiment_name_padding: int = 0,
        plot_experiment_info: bool = True,
        precision: int = 4,
    ) -> matplotlib.figure.Figure:
        # Typehint the metric and class_label variables
        metric: MetricLike
        class_label: int

        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        # Import optional dependencies
        try:
            import tabulate
            import matplotlib
            import matplotlib.pyplot as plt

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Visualization requires optional dependencies: [matplotlib, pyplot]. Currently missing: {e}"
            )

        if axis_fontsize is None:
            axis_fontsize = fontsize

        # Figure out a decent default figsize
        total_num_experiments = (
            len(self._list_experiments()) + 2 * len(self.experiments) + 1
        )

        if figsize is None:
            # Try to set a decent default figure size
            _figsize = (6.3, max(0.28 * total_num_experiments, 2.5))
        else:
            _figsize = figsize

        # Get the height ratios of each experiment group subplot
        heights = [
            len(experiment_group.experiments) + agg_offset + 1
            for experiment_group in self._experiment_store.values()
        ]

        heights[0] += 1

        fig, axes = plt.subplots(
            nrows=len(self._experiment_store),
            ncols=1,
            sharex=True,
            sharey=False,
            height_ratios=heights,
            figsize=_figsize,
        )
        if isinstance(axes, matplotlib.axes._axes.Axes):  # type: ignore
            axes = np.array([axes])

        for i, (experiment_group_name, experiment_group) in enumerate(
            self._experiment_store.items()
        ):
            num_experiments = 0
            all_experiment_names = []
            all_summaries = []
            # Get boxpstats for each individual experiment =============================
            for ii, (experiment_name, experiment) in enumerate(
                experiment_group.experiments.items()
            ):
                all_experiment_names.append(experiment_name)

                samples = self.get_metric_samples(
                    metric=metric.name,
                    experiment_name=f"{experiment_group_name}/{experiment_name}",
                    sampling_method=SamplingMethod.POSTERIOR,
                ).values[:, class_label]

                summary = summarize_posterior(
                    posterior_samples=samples, ci_probability=self._ci_probability
                )

                all_summaries.append(summary)

                # Median
                axes[i].scatter(
                    x=summary.median,
                    y=ii,
                    marker=median_marker,
                    facecolor=median_marker_face_colour,
                    edgecolor=median_marker_edge_colour,
                    s=median_marker_size**2,
                    linewidth=median_marker_line_width,
                    zorder=1,
                )

                # HDI Lines
                axes[i].hlines(
                    xmin=summary.hdi[0],
                    xmax=summary.hdi[1],
                    y=ii,
                    zorder=0,
                    linewidth=hdi_lines_width,
                    color=hdi_lines_colour,
                    ls=hdi_lines_format,
                )

                num_experiments += 1

            # Get boxp stats for aggregated distribution ===============================
            aggregation_result: ExperimentAggregationResult = self.get_metric_samples(
                metric=metric.name,
                experiment_name=f"{experiment_group_name}/aggregated",
                sampling_method=SamplingMethod.POSTERIOR,
            )  # type: ignore

            samples = aggregation_result.values[:, class_label]

            summary = summarize_posterior(
                posterior_samples=samples, ci_probability=self._ci_probability
            )

            # Median
            axes[i].scatter(
                x=summary.median,
                y=num_experiments + agg_offset,
                marker=agg_median_marker,
                facecolor=agg_median_marker_face_colour,
                edgecolor=agg_median_marker_edge_colour,
                s=agg_median_marker_size**2,
                linewidth=agg_median_marker_line_width,
                zorder=1,
            )

            # HDI Lines
            axes[i].hlines(
                xmin=summary.hdi[0],
                xmax=summary.hdi[1],
                y=num_experiments + agg_offset,
                zorder=0,
                linewidth=hdi_lines_width,
                color=hdi_lines_colour,
                ls=hdi_lines_format,
            )

            if plot_agg_median_line:
                axes[i].axvline(
                    summary.median,
                    *axes[i].get_ylim(),
                    ls=agg_median_line_format,
                    c=agg_median_line_colour,
                    linewidth=agg_median_line_width,
                    zorder=0,
                )

            # Add the ytick locations ==================================================
            axes[i].set_ylim(-1 if i == 0 else -0.5, num_experiments + agg_offset + 0.5)
            axes[i].set_yticks(
                range(-1 if i == 0 else 0, num_experiments + agg_offset + 1)
            )

            # Invert the axis
            axes[i].invert_yaxis()

            # Add experiment labels to left axis =======================================
            if plot_experiment_name:
                aggregator_name = self._metric_to_aggregator[metric].name

                longest_experiment_name = max(
                    15,
                    len(aggregator_name),
                    max(map(len, all_experiment_names)),
                )

                all_experiment_labels = (
                    (
                        [
                            f"{'Experiment Name':<{longest_experiment_name}}{'':{experiment_name_padding}}"
                        ]
                        if i == 0
                        else []
                    )
                    + [
                        f"{experiment_name[:longest_experiment_name]:<{longest_experiment_name}}{'':{experiment_name_padding}}"
                        for experiment_name in all_experiment_names
                    ]
                    + [""] * agg_offset
                    + [
                        f"{aggregator_name:<{longest_experiment_name}}{'':{experiment_name_padding}}"
                        + f"\nI2={fmt(aggregation_result.heterogeneity_results[class_label].i2, precision=precision, mode='%'):<{longest_experiment_name}}{'':{experiment_name_padding}}"
                    ]
                )

                # Set the labels
                axes[i].set_yticklabels(
                    all_experiment_labels,
                    fontsize=fontsize,
                    fontname=fontname,
                )
            else:
                axes[i].set_yticklabels(["" for _ in axes[i].get_yticks()])

            # Remove the ticks
            axes[i].tick_params(axis="y", which="both", length=0)

            axes[i].tick_params(axis="x", which="both", labelsize=axis_fontsize)

            # Clone the axis ===========================================================
            ax_clone = axes[i].twinx()

            ax_clone.invert_yaxis()

            # Add the yticks to match the original axis
            ax_clone.set_yticks(axes[i].get_yticks())

            ax_clone.set_ylim(axes[i].get_ylim())

            # Remove the y-ticks
            ax_clone.tick_params(axis="y", which="both", length=0)
            ax_clone.tick_params(axis="x", which="both", labelsize=axis_fontsize)

            # Add experiment summary info ==========================================
            if plot_experiment_info:

                def summary_to_row(summary):
                    if summary.hdi[1] - summary.hdi[0] > 1e-4:
                        hdi_str = f"[{fmt(summary.hdi[0], precision=precision, mode='f')}, {fmt(summary.hdi[1], precision=precision, mode='f')}]"
                    else:
                        hdi_str = f"[{fmt(summary.hdi[0], precision=precision, mode='e')}, {fmt(summary.hdi[1], precision=precision, mode='e')}]"

                    return {
                        "Median": summary.median,
                        f"{summary.headers[2]}": hdi_str,
                        "MU": summary.hdi[1] - summary.hdi[0],
                    }

                summary_rows = [summary_to_row(s) for s in all_summaries + [summary]]

                tabulate_str = tabulate.tabulate(  # type: ignore
                    tabular_data=summary_rows,
                    headers="keys",
                    colalign=["right"] * 3,
                    floatfmt=f".{precision}f",
                    tablefmt="plain",
                )

                tabulate_rows = tabulate_str.split("\n")

                all_experiment_info = (
                    ([tabulate_rows[0]] if i == 0 else [])
                    + tabulate_rows[1 : num_experiments + 1]
                    + [""] * agg_offset
                    + [tabulate_rows[-1]]
                )

                ax_clone.set_yticklabels(
                    all_experiment_info,
                    fontsize=fontsize,
                    fontname=fontname,
                )
            else:
                ax_clone.set_yticklabels(["" for _ in ax_clone.get_yticks()])

            # Remove spines ============================================================
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["left"].set_visible(False)

            ax_clone.spines["right"].set_visible(False)
            ax_clone.spines["top"].set_visible(False)
            ax_clone.spines["left"].set_visible(False)

            # Metric group label ===============================================
            axes[i].set_ylabel(experiment_group_name, fontsize=fontsize)

        # Metric label =========================================================
        axes[-1].set_xlabel(metric.full_name, fontsize=fontsize)

        fig.subplots_adjust()
        fig.tight_layout()

        return fig
