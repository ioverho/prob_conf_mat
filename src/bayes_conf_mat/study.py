import os
import typing
import uuid
import warnings
from pathlib import Path
from functools import cached_property
from collections import OrderedDict

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.utils.wrappers import changes_state
from bayes_conf_mat.utils.cache import InMemoryCache, PickleCache
from bayes_conf_mat.experiment_manager import ExperimentManager
from bayes_conf_mat.metrics import (
    MetricCollection,
    Metric,
    AveragedMetric,
)
from bayes_conf_mat.io import get_io
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator
from bayes_conf_mat.reporting import (
    experiment_summaries,
    random_experiment_summaries,
    pairwise_comparison_to_random,
    pairwise_comparison,
)
from bayes_conf_mat.reporting.visualization import (
    distribution_plot,
    diff_distribution_plot,
)

# from bayes_conf_mat.report.utils import (
#    aggregation_summary_table,
#    forest_plot,
#    pairwise_comparison_plot,
#    listwise_comparison_table,
#    expected_reward_table,
#    values_to_table_row,
# )


class Study:
    def __init__(
        self,
        name: typing.Optional[str] = None,
        seed: int | np.random.BitGenerator | None = None,
        num_samples: typing.Optional[int] = None,
        ci_probability: typing.Optional[float] = None,
        metrics: typing.Optional[typing.Iterable[str]] = (),
        cache_dir: str | None = None,
        overwrite: bool = False,
    ):
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

        # The study's RNG
        if seed is None:
            self.rng = np.random.default_rng()
        elif isinstance(seed, int) or isinstance(seed, float):
            self.rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.BitGenerator) or isinstance(
            seed, np.random.Generator
        ):
            self.rng = seed

        self.num_samples = num_samples

        self.ci_probability = ci_probability

        # The experiment group store
        self.experiment_groups = OrderedDict()

        # The collection of metrics
        self.metrics = MetricCollection(metrics)

        # Slots for the config to be stored
        self.config = None
        self._yaml_config = None

        # Check if we're allowed to cache results
        # And where that caching should happen
        self.overwrite = overwrite

        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir).resolve()

            os.makedirs(self.cache_dir, exist_ok=True)

            self.cache_location = self.cache_dir / self.name

            if self.cache_location.exists():
                if not overwrite:
                    raise ValueError(
                        f"Report location exists, and overwrite is False: {self.cache_location}"
                    )
                else:
                    warnings.warn(
                        message=f"Report location exists! Overwriting: {self.cache_location}"
                    )
            else:
                os.makedirs(self.cache_location, exist_ok=False)

            self.cache = PickleCache()

        else:
            self.cache_location = None
            self.cache = InMemoryCache()

        # The experiment group store
        self.experiment_groups = dict()

        # The experiment aggregators store
        self.metric_to_aggregator = dict()

    def __repr__(self):
        return f"Study({self.name}, experiment_groups={list(self.experiment_groups.keys())}), metrics={self.metrics}"

    def __str__(self):
        return f"Study({self.name}, experiment_groups={list(self.experiment_groups.keys())}, metrics={self.metrics})"

    def __len__(self) -> int:
        return len(self.experiment_groups)

    # TODO: document this method
    @changes_state
    def add_experiment(
        self,
        name: str,
        confusion_matrix: typing.Dict[str, typing.Any] | np.ndarray,
        prevalence_prior: float | str | jtyping.Float[np.ndarray, " num_classes"] = 0.0,
        confusion_prior: float | str | jtyping.Float[np.ndarray, " num_classes"] = 0.0,
        **kwargs,
    ):
        split_name = name.split("/")

        if len(split_name) == 2:
            experiment_group_name = split_name[0]
            experiment_name = split_name[1]

        elif len(split_name) == 1:
            experiment_group_name = split_name[0]
            experiment_name = split_name[0]

            warnings.warn(
                f"Received experiment without experiment group: {experiment_name}. Adding to its own experiment group. To specify an experiment group, pass a string formatted as 'group\\name'."
            )

        elif len(split_name) >= 2:
            raise NotImplementedError(
                f"Received invalid experiment name. Currently: {name}. Must have at most 1 '/' character. Hierarchical experiment groups not (yet) implemented."
            )

        else:
            raise ValueError("Something went wrong here...")

        # Wrap the confusion matrix if it is a np.ndarray
        if isinstance(confusion_matrix, np.ndarray):
            confusion_matrix = get_io(format="in_memory", data=confusion_matrix).load()

        # Get the experiment group if it exists, otherwise create it
        if experiment_group_name not in self.experiment_groups:
            # Give the new experiment group its own RNG
            # Should be independent from the study's RNG and all other experiment groups' RNGs
            indep_rng = self.rng.spawn(1)[0]

            experiment_group = ExperimentManager(
                name=experiment_group_name,
                seed=indep_rng,
                **kwargs,
            )

            experiment_group.metrics = self.metrics

            self.experiment_groups[experiment_group_name] = experiment_group

        # Finally, add the experiment to the right experiment group
        experiment_group = self.experiment_groups[experiment_group_name].add_experiment(
            name=experiment_name,
            confusion_matrix=confusion_matrix,
            prevalence_prior=prevalence_prior,
            confusion_prior=confusion_prior,
        )

    @changes_state
    def add_metric(
        self,
        metric: str
        | typing.Type[Metric]
        | typing.Type[AveragedMetric]
        | typing.Iterable[str | typing.Type[Metric] | typing.Type[AveragedMetric]],
    ) -> None:
        self.metrics.add(metric)

    @changes_state
    def add_experiment_aggregation(
        self, metric_name: str, aggregation_config: typing.Dict[str, typing.Any]
    ) -> None:
        indep_rng = self.rng.spawn(1)[0]

        aggregator = get_experiment_aggregator(rng=indep_rng, **aggregation_config)

        metric = self.metrics[metric_name]

        self.metric_to_aggregator[metric] = aggregator

    def clean_cache(self):
        self.cache.clean()

    @cached_property
    def num_classes(self):
        all_num_classes = set()
        for experiment_group in self.experiment_groups.values():
            all_num_classes.add(experiment_group.num_classes)

        if len(all_num_classes) > 1:
            raise ValueError(
                f"Inconsistent number of classes in experiment groups: {all_num_classes}"
            )
        else:
            return list(all_num_classes)[0]

    def _validate_metric_class_label_combination(
        self, metric: Metric | AveragedMetric, class_label: int
    ):
        try:
            metric = self.metrics[metric]
        except KeyError:
            raise KeyError(
                f"Could not find metric '{metric}' in the metrics collection. Consider adding it using `Study.add_metric`"
            )

        if metric.is_multiclass:
            if not ((class_label == 0) or (class_label is None)):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0
        else:
            if class_label is None:
                raise ValueError(
                    f"Metric '{metric.name}' is not multiclass. You must provide a class label."
                )
            elif class_label < 0 or class_label > self.num_classes - 1:
                raise ValueError(
                    f"Class label must be in range [0, {self.num_classes - 1}]. Currently {class_label}."
                )

        return metric, class_label

    def _sample_metrics(self, sampling_method: str):
        if sampling_method in ["posterior", "prior", "random"]:
            # Compute metrics for the entire experiment group
            for experiment_group in self.experiment_groups.values():
                experiment_group_results = experiment_group.sample_metrics(
                    metrics=self.metrics,
                    sampling_method=sampling_method,
                    num_samples=self.num_samples,
                    metric_to_aggregator=self.metric_to_aggregator,
                )

                # Cache the aggregated results
                for (
                    metric,
                    aggregation_result,
                ) in experiment_group_results.aggregation_result.items():
                    self.cache.cache(
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
                            keys=[
                                metric.name,
                                experiment_group.name,
                                experiment.name,
                                sampling_method,
                            ],
                            value=experiment_result,
                        )

        elif sampling_method in ["input"]:
            # Compute metrics for the entire experiment group
            for experiment_group in self.experiment_groups.values():
                for experiment in experiment_group.experiments.values():
                    experiment_results = experiment.sample_metrics(
                        metrics=self.metrics,
                        sampling_method="input",
                        num_samples=self.num_samples,
                    )

                    for metric, experiment_result in experiment_results.items():
                        self.cache.cache(
                            keys=[
                                metric.name,
                                experiment_group.name,
                                experiment.name,
                                "input",
                            ],
                            value=experiment_result,
                        )

        else:
            raise NotImplementedError(
                f"Sampling method '{sampling_method}' not implemented for `Study._sample_metrics`"
            )

    def get_metric_samples(
        self,
        metric: str,
        experiment_group: str,
        sampling_method: str,
        experiment: str | typing.Literal["aggregated"],
    ):
        keys = [metric, experiment_group, experiment, sampling_method]

        if keys in self.cache:
            result = self.cache.load(keys)

        else:
            self._sample_metrics(sampling_method=sampling_method)

            result = self.cache.load(keys)

        return result

    # TODO: document this method
    def report_metric_summaries(
        self,
        metric: str,
        class_label: int | None = None,
        table_fmt: str = "html",
        precision: int = 4,
        **tabulate_kwargs,
    ) -> str:
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        summary_table = experiment_summaries(
            study=self,
            metric=metric,
            class_label=class_label,
            table_fmt=table_fmt,
            precision=precision,
            **tabulate_kwargs,
        )

        return summary_table

    def report_random_metric_summaries(
        self,
        metric: str,
        class_label: int | None = None,
        table_fmt: str = "html",
        precision: int = 4,
        **tabulate_kwargs,
    ) -> str:
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        summary_table = random_experiment_summaries(
            study=self,
            metric=metric,
            class_label=class_label,
            table_fmt=table_fmt,
            precision=precision,
            **tabulate_kwargs,
        )

        return summary_table

    def plot_metric_summaries(
        self,
        metric: str,
        class_label: int | None = None,
        **kwargs,
    ):
        """Plots the distrbution of sampled metric values for a particular metric and class combination.

        Args:
            metric (str): the name of the metric
            class_label (int | None, optional): the class label. Defaults to None.
            observed_values (typing.Dict[str, ExperimentResult]): the observed metric values
            sampled_values (typing.Dict[str, ExperimentResult]): the sampled metric values
            metric (Metric | AveragedMetric): the metric
            method (str, optional): the method for displaying a histogram, provided by Seaborn. Can be either a histogram or KDE. Defaults to "kde".
            bandwidth (float, optional): the bandwith parameter for the KDE. Corresponds to [Seaborn's `bw_adjust` parameter](https://seaborn.pydata.org/generated/seaborn.kdeplot.html). Defaults to 1.0.
            bins (int | typing.List[int] | str, optional): the number of bins to use in the histrogram. Corresponds to [numpy's `bins` parameter](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges). Defaults to "auto".
            normalize (bool, optional): if normalized, each distribution will be scaled to [0, 1]. Otherwise, uses a shared y-axis. Defaults to False.
            figsize (typing.Tuple[float, float], optional): the figure size, in inches. Corresponds to matplotlib's `figsize` parameter. Defaults to None, in which case a decent default value will be approximated.
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
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric, class_label=class_label
        )

        fig = distribution_plot(
            study=self,
            metric=metric,
            class_label=class_label,
            **kwargs,
        )

        return fig

    def report_comparison_to_random(
        self,
        metric: str,
        class_label: int,
        min_sig_diff: typing.Optional[float] = None,
        precision: int = 4,
        tablefmt: str = "html",
        **tabulate_kwargs,
    ) -> str:
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric,
            class_label=class_label,
        )

        summary_table = pairwise_comparison_to_random(
            self,
            metric=metric,
            class_label=class_label,
            min_sig_diff=min_sig_diff,
            precision=precision,
            tablefmt=tablefmt,
            **tabulate_kwargs,
        )

        return summary_table

    def report_comparison(
        self,
        metric: str,
        class_label: int,
        experiment_group_a: str,
        experiment_group_b: str,
        experiment_a: str | typing.Literal["aggregated"] = "aggregated",
        experiment_b: str | typing.Literal["aggregated"] = "aggregated",
        min_sig_diff: typing.Optional[float] = None,
        precision: int = 4,
    ) -> None:
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric,
            class_label=class_label,
        )

        pairwise_comparison(
            study=self,
            metric=metric,
            class_label=class_label,
            experiment_group_a=experiment_group_a,
            experiment_group_b=experiment_group_b,
            experiment_a=experiment_a,
            experiment_b=experiment_b,
            min_sig_diff=min_sig_diff,
            precision=precision,
        )

    def plot_comparison(
        self,
        metric: str,
        class_label: int,
        experiment_group_a: str,
        experiment_group_b: str,
        experiment_a: str | typing.Literal["aggregated"] = "aggregated",
        experiment_b: str | typing.Literal["aggregated"] = "aggregated",
        min_sig_diff: typing.Optional[float] = None,
        **kwargs,
    ) -> str:
        metric, class_label = self._validate_metric_class_label_combination(
            metric=metric,
            class_label=class_label,
        )

        fig = diff_distribution_plot(
            study=self,
            metric=metric,
            class_label=class_label,
            experiment_group_a=experiment_group_a,
            experiment_group_b=experiment_group_b,
            experiment_a=experiment_a,
            experiment_b=experiment_b,
            min_sig_diff=min_sig_diff,
            **kwargs,
        )

        return fig


# Deprecated code
"""

    def _parse_config(self, config: str | Config, encoding: str = "utf-8"):
        raise NotImplementedError
        # if isinstance(config, Config):
        #    self.config = config
        # elif isinstance(config, str) or isinstance(config, Path):
        #    if Path(config).is_file():
        #        self.config = load_config_from_file(
        #            config_location=config, encoding=encoding
        #        )
        #    else:
        #        try:
        #            self.config = load_config_from_text(config_string=config)
        #        except Exception as e:
        #            raise ValueError(f"Error when parsing config as config string: {e}")
        # else:
        #    raise ValueError(f"Type of config variable not implemented: {type(config)}")

    @classmethod
    def from_config(
        cls, config: str | Path | Config, encoding: str = "utf-8", **init_kwargs
    ):
        instance = cls(name=None, seed=None, num_samples=None, **init_kwargs)

        instance._parse_config(config=config, encoding=encoding)

        instance._name = instance.config.name
        instance.seed = instance.config.seed
        instance.num_samples = instance.config.num_samples

        for experiment_group_name in instance.config.experiments.keys():
            for experiment_name, experiment_config in instance.config.experiments[
                experiment_group_name
            ].items():
                instance.add_experiment(
                    name=f"{experiment_group_name}/{experiment_name}",
                    confusion_matrix=experiment_config,
                    prevalence_prior=instance.config.prevalence_prior,
                    confusion_prior=instance.config.confusion_prior,
                )

        for metric, aggregation_dict in instance.config.metrics.items():
            instance.add_metric(metric)
            instance.add_experiment_aggregation(metric, aggregation_dict)

        return instance

    def report_experiment_aggregation(
        self,
        metric_name: str,
        class_label: int,
        experiment_group_name: str,
    ):
        experiment_group = self.experiment_groups[experiment_group_name]
        metric = experiment_group.metrics[metric_name]

        if metric.is_multiclass:
            if not ((class_label == 0) or (class_label is None)):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0

        point_estimates = self._sample_metrics(
            sampling_method="input",
            experiment_group=experiment_group,
            metric_name=metric_name,
        )[class_label]

        metric_values = self._sample_metrics(
            sampling_method="posterior",
            experiment_group=experiment_group,
            metric_name=metric_name,
        )[class_label]

        agg_metric_values = self._sample_agg_metrics(
            sampling_method="posterior",
            experiment_group=experiment_group,
            metric_name=metric_name,
        )[class_label]

        aggregation_summary = ""

        aggregation_summary += "```yaml\n"
        aggregation_summary += "\n".join(
            map(
                lambda x: f"{x[0]}: {repr(x[1])}",
                self.config.metrics[metric_name]._attrs.items(),
            )
        )
        aggregation_summary += "\n```"

        aggregation_summary += "\n\n"

        aggregation_summary += aggregation_summary_table(
            point_estimates=point_estimates,
            individual_results=metric_values,
            aggregated_results=agg_metric_values,
            ci_probability=self.config.ci_probability,
            table_fmt="github",
        )

        aggregation_summary += "\n\n"

        aggregation_summary += agg_metric_values.heterogeneity.template_sentence()

        return aggregation_summary

    def report_forest_plot(
        self,
        metric_name: str,
        experiment_group_name: str,
        class_label: int = None,
        precision: int = 4,
        fontsize: typing.Optional[int] = 9,
        figsize: typing.Optional[typing.Tuple[int, int]] = None,
        add_summary_info: bool = True,
        agg_offset: typing.Optional[int] = 1,
        max_hist_height: float = 0.7,
    ):
        experiment_group = self.experiment_groups[experiment_group_name]
        metric = experiment_group.metrics[metric_name]

        if metric.is_multiclass:
            if not ((class_label == 0) or (class_label is None)):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0

        metric_values = self._sample_metrics(
            sampling_method="posterior",
            experiment_group=experiment_group,
            metric_name=metric_name,
        )[class_label]

        agg_metric_values = self._sample_agg_metrics(
            sampling_method="posterior",
            experiment_group=experiment_group,
            metric_name=metric_name,
        )[class_label]

        fp_fig = forest_plot(
            individual_samples=metric_values,
            aggregated_samples=[agg_metric_values],
            bounds=metric.bounds,
            ci_probability=self.config.ci_probability,
            precision=precision,
            fontsize=fontsize,
            figsize=figsize,
            add_summary_info=add_summary_info,
            agg_offset=agg_offset,
            max_hist_height=max_hist_height,
        )

        return fp_fig

    def _pairwise_compare(
        self,
        metric_name: str,
        class_label: int,
        experiment_group_name_a: str,
        experiment_group_name_b: str,
        min_sig_diff: float = None,
    ):
        if experiment_group_name_a == experiment_group_name_b:
            raise ValueError("Experiment 'a' and 'b' point to the experiment.")

        metric = self.experiment_groups[experiment_group_name_a].metrics[metric_name]

        if metric.is_multiclass:
            if (class_label != 0) and (class_label is not None):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0

        result_a = self._sample_agg_metrics(
            sampling_method="posterior",
            experiment_group=self.experiment_groups[experiment_group_name_a],
            metric_name=metric_name,
        )[class_label]

        result_b = self._sample_agg_metrics(
            sampling_method="posterior",
            experiment_group=self.experiment_groups[experiment_group_name_b],
            metric_name=metric_name,
        )[class_label]

        comparison_result = pairwise_compare(
            result_a,
            result_b,
            ci_probability=self.config.ci_probability,
            min_sig_diff=min_sig_diff,
            lhs_name=experiment_group_name_a,
            rhs_name=experiment_group_name_b,
        )

        return comparison_result

    def report_pairwise_comparison(
        self,
        metric_name: str,
        experiment_group_name_a: str,
        experiment_group_name_b: str,
        class_label: int = None,
        min_sig_diff: float = None,
        precision: int = 4,
    ):
        comparison_result = self._pairwise_compare(
            metric_name=metric_name,
            class_label=class_label,
            experiment_group_name_a=experiment_group_name_a,
            experiment_group_name_b=experiment_group_name_b,
            min_sig_diff=min_sig_diff,
        )

        return comparison_result.template_sentence(precision=precision)

    def report_pairwise_comparison_plot(
        self,
        metric_name: str,
        experiment_group_name_a: str,
        experiment_group_name_b: str,
        class_label: int = None,
        min_sig_diff: float = None,
        precision: int = 4,
        figsize: typing.Tuple[float, float] = None,
    ):
        comparison_result = self._pairwise_compare(
            metric_name=metric_name,
            class_label=class_label,
            experiment_group_name_a=experiment_group_name_a,
            experiment_group_name_b=experiment_group_name_b,
            min_sig_diff=min_sig_diff,
        )

        fig = pairwise_comparison_plot(
            comparison_result, precision=precision, figsize=figsize
        )

        return fig

    def _pairwise_compare_random(
        self,
        metric_name: str,
        experiment_group_name: str,
        class_label: int = None,
        min_sig_diff: float = None,
    ):
        metric = self.experiment_groups[experiment_group_name].metrics[metric_name]

        if metric.is_multiclass:
            if (class_label != 0) and (class_label is not None):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0

        actual_result = self._sample_agg_metrics(
            sampling_method="posterior",
            experiment_group=self.experiment_groups[experiment_group_name],
            metric_name=metric_name,
        )[class_label]

        random_results = self._sample_agg_metrics(
            sampling_method="random",
            experiment_group=self.experiment_groups[experiment_group_name],
            metric_name=metric_name,
        )[class_label]

        comparison_result = pairwise_compare(
            actual_result,
            random_results,
            ci_probability=self.config.ci_probability,
            min_sig_diff=min_sig_diff,
            lhs_name=experiment_group_name,
            rhs_name="random",
        )

        return comparison_result

    def report_pairwise_random_comparison(
        self,
        metric_name: str,
        experiment_group_name: str,
        class_label: int = None,
        min_sig_diff: float = None,
        precision: int = 4,
    ):
        comparison_result = self._pairwise_compare_random(
            metric_name=metric_name,
            class_label=class_label,
            experiment_group_name=experiment_group_name,
            min_sig_diff=min_sig_diff,
        )

        return comparison_result.template_sentence(precision=precision)

    def _listwise_compare(self, metric_name: str, class_label: int = None):
        metric = list(self.experiment_groups.values())[0].metrics[metric_name]

        if metric.is_multiclass:
            if not ((class_label == 0) or (class_label is None)):
                warnings.warn("Metric is multiclass, ignoring class label.")

            class_label = 0

        experiment_values = [
            self._sample_agg_metrics(
                sampling_method="posterior",
                experiment_group=experiment_group,
                metric_name=metric_name,
            )[class_label]
            for experiment_group in self.experiment_groups.values()
        ]

        listwise_comparison_result = listwise_comparison(
            experiment_values=experiment_values,
            experiment_names=list(
                map(lambda x: x.experiment_group.name, experiment_values)
            ),
            metric_name=metric_name,
        )

        return listwise_comparison_result

    def report_listwise_comparison(
        self, metric_name: str, class_label: int = None, precision: int = 4
    ):
        listwise_comparison_result = self._listwise_compare(
            metric_name=metric_name, class_label=class_label
        )

        return listwise_comparison_table(
            listwise_comparison_result=listwise_comparison_result,
            precision=precision,
        )

    def report_expected_reward(
        self,
        metric_name: str,
        class_label: int = None,
        rewards: typing.Optional[typing.List[float]] = None,
        precision: int = 4,
    ):
        listwise_comparison_result = self._listwise_compare(
            metric_name=metric_name, class_label=class_label
        )

        p_rank_given_experiment = listwise_comparison_result.p_rank_given_experiment

        # Use the mean-reciprocal rank if no rewards provided
        if rewards is None:
            rewards = 1 / (np.arange(stop=p_rank_given_experiment.shape[0]) + 1)

        # Otherwise handle a list of rewards
        else:
            if len(rewards) > p_rank_given_experiment.shape[0]:
                raise ValueError(
                    "Rewards list is longer than the number of experiments."
                )

            # Pad the rewards list wth 0s if too short
            rewards = np.pad(
                rewards,
                pad_width=(0, p_rank_given_experiment.shape[0] - len(rewards)),
                mode="constant",
                constant_values=0.0,
            )

        expected_reward = p_rank_given_experiment @ rewards

        reward_table = expected_reward_table(
            expected_reward=expected_reward,
            names=listwise_comparison_result.experiment_names,
            precision=precision,
        )

        return reward_table
"""
