import os
import typing
import uuid
import warnings
from pathlib import Path

import numpy as np

from bayes_conf_mat.config import load_config, Config
from bayes_conf_mat.experiment import ExperimentResult
from bayes_conf_mat.experiment_manager import ExperimentManager
from bayes_conf_mat.significance_testing import pairwise_compare, listwise_comparison
from bayes_conf_mat.report.utils import (
    aggregation_summary_table,
    forest_plot,
    pairwise_comparison_plot,
    listwise_comparison_table,
    expected_reward_table,
)
from bayes_conf_mat.utils.cache import InMemoryCache, PickleCache


class Study:
    def __init__(
        self,
        cache_dir: str | None = None,
        overwrite: bool = False,
    ):
        # Slots for the config to be stored
        self.config = None
        self._yaml_config = None
        self._name = str(uuid.uuid4())

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

    def __repr__(self):
        return f"Study({self.name}, experiment_groups=f{list(self.experiment_groups.keys())})"

    def __str__(self):
        return f"Study({self.name}, experiment_groups=f{list(self.experiment_groups.keys())})"

    def __len__(self) -> int:
        return len(self.experiment_groups)

    def clean_cache(self):
        self.cache.clean()

    @property
    def name(self):
        if self.config is None:
            return self._name
        else:
            return self.config.name

    def _parse_config(self, config: str | Config, encoding: str = "utf-8"):
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = load_config(config_location=config, encoding=encoding)

    @classmethod
    def from_config(cls, config: str | Config, encoding: str = "utf-8", **init_kwargs):
        instance = cls(**init_kwargs)

        instance._parse_config(config=config, encoding=encoding)

        # Equip the different experiment groups with an *independent* RNG ======
        # Spawn the RNG for the study as a whole
        study_rng = np.random.default_rng(instance.config["seed"])

        # Spawn independent RNGs for each experiment group
        # This ensures the synthetic confusion matrices aren't correlated
        indep_rngs = study_rng.spawn(len(instance.config.experiments))

        # The experiment group store
        instance.experiment_groups = dict()
        # Build the experiment groups
        instance.experiment_groups = dict()
        for (name, experiments), new_rng in zip(
            instance.config.experiments.items(), indep_rngs
        ):
            experiment_group = ExperimentManager(
                name=name,
                experiments=experiments,
                num_samples=instance.config.num_samples,
                seed=new_rng,
                prevalence_prior=instance.config.prevalence_prior,
                confusion_prior=instance.config.confusion_prior,
                metrics=instance.config.metrics.keys(),
                experiment_aggregations=instance.config.metrics,
            )

            instance.experiment_groups[name] = experiment_group

        # TODO: allow defining cache from config

        return instance

    def _sample_metrics(
        self,
        sampling_method: str,
        experiment_group: ExperimentManager,
        metric_name: typing.Optional[str] = None,
    ) -> typing.Iterator[
        typing.Tuple[ExperimentManager, typing.Dict[str, typing.List[ExperimentResult]]]
    ]:
        # Try to load from cache
        cache_result = self.cache.load(
            ["metric_results", sampling_method, experiment_group.name]
            + ([] if metric_name is None else [metric_name]),
            default=None,
        )

        if cache_result is None:
            # If we haven't cached these results yet, compute them and cache
            metric_results = experiment_group.compute_metrics(
                sampling_method=sampling_method
            )

            self.cache.cache(
                ["metric_results", sampling_method, experiment_group.name],
                value=metric_results,
            )

            if metric_name is not None:
                metric_results = metric_results[metric_name]

        else:
            # Otherwise return the cached results
            metric_results = cache_result

        return metric_results

    def _sample_agg_metrics(
        self,
        sampling_method: str,
        experiment_group: ExperimentManager,
        metric_name: typing.Optional[str] = None,
    ) -> typing.Iterator[
        typing.Tuple[ExperimentManager, typing.Dict[str, typing.List[ExperimentResult]]]
    ]:
        # Try to load from cache
        cache_result = self.cache.load(
            ["agg_metric_results", sampling_method, experiment_group.name]
            + ([] if metric_name is None else [metric_name]),
            default=None,
        )

        if cache_result is None:
            # If we haven't cached these results yet, compute them and cache

            # First load or compute the metric results (not aggregated)
            metric_results = self._sample_metrics(
                sampling_method=sampling_method,
                experiment_group=experiment_group,
            )

            agg_metric_results = experiment_group.aggregate_experiments(metric_results)

            self.cache.cache(
                ["agg_metric_results", sampling_method, experiment_group.name],
                value=agg_metric_results,
            )

            if metric_name is not None:
                agg_metric_results = agg_metric_results[metric_name]

        else:
            # Otherwise return the cached results
            agg_metric_results = cache_result

        return agg_metric_results

    def sample_metrics(
        self,
        sampling_method: str,
        aggregated: bool = False,
    ) -> typing.Iterator[
        typing.Tuple[ExperimentManager, typing.Dict[str, typing.List[ExperimentResult]]]
    ]:
        # Iterate over the experiment groups
        for experiment_group_name, experiment_group in self.experiment_groups.items():
            if aggregated:
                result = self._sample_agg_metrics(
                    sampling_method=sampling_method,
                    experiment_group=experiment_group,
                )
            else:
                result = self._sample_metrics(
                    sampling_method=sampling_method,
                    experiment_group=experiment_group,
                )

            yield experiment_group, result

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
