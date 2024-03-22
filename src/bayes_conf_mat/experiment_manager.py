import typing
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment import Experiment, ExperimentResult
from bayes_conf_mat.metrics import (
    Metric,
    AveragedMetric,
    RootMetric,
    MetricCollection,
)
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator
from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.experiment_aggregation.utils.heterogeneity import (
    HeterogeneityResult,
)


# TODO: document class
class ExperimentManager:
    def __init__(
        self,
        name: str,
        experiments: typing.Dict[
            str, typing.Dict | jtyping.Int[np.ndarray, " num_classes num_classes"]
        ],
        num_samples: typing.Optional[int],
        seed: typing.Optional[int | np.random.BitGenerator],
        prevalence_prior: str | int | jtyping.Int[np.ndarray, " num_classes"],
        confusion_prior: str
        | int
        | jtyping.Int[np.ndarray, " num_classes num_classes"],
        metrics: typing.Optional[typing.Iterable[str]] = (),
        experiment_aggregations: typing.Optional[
            typing.Dict[str, typing.Dict[str, str]]
        ] = None,
    ) -> None:
        self.name = name

        # Import hyperparameters
        # The number of synthetic confusion matrices to sample
        self.num_samples = num_samples

        # The prior strategy to use for each experiment
        self.prevalence_prior = prevalence_prior
        self.confusion_prior = confusion_prior

        # The manager's RNG
        if isinstance(seed, int) or isinstance(seed, float):
            self.rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.BitGenerator) or isinstance(
            seed, np.random.Generator
        ):
            self.rng = seed

        # The collection of experiments
        self.num_classes = None
        self.experiments = OrderedDict()
        self.add_experiments(experiments)

        # The collection of metrics
        self.metrics = MetricCollection(metrics)

        # The collection of experiment aggregators
        self.experiment_aggregations = dict()
        self.metric_to_aggregator = dict()
        if experiment_aggregations is not None:
            for metric_name, aggregation_config in experiment_aggregations.items():
                self.add_experiment_aggregation(
                    metric_name=metric_name, aggregation_config=aggregation_config
                )

    @property
    def num_experiments(self):
        return len(self.experiments)

    # TODO: document method
    def add_experiments(
        self,
        experiments: typing.Dict[
            str, typing.Dict | jtyping.Int[np.ndarray, " num_classes num_classes"]
        ],
    ) -> None:
        # Each experiment gets its own RNG, spawned from the manager's RNG
        # In theory, should allow for parallelizing the different experiments
        indep_rngs = self.rng.spawn(len(experiments))

        temp_experiments = OrderedDict()
        for (name, confusion_matrix), rng in zip(experiments.items(), indep_rngs):
            temp_experiments[name] = Experiment(
                # Provided for the user
                # Unique to each experiment
                name=name,
                confusion_matrix=confusion_matrix,
                # Shared across experiments
                prevalence_prior=self.prevalence_prior,
                confusion_prior=self.confusion_prior,
                # Provided by the manager
                rng=rng,
            )

            if self.num_classes is None:
                self.num_classes = temp_experiments[name].num_classes
            else:
                experiment_num_classes = temp_experiments[name].num_classes
                if experiment_num_classes != self.num_classes:
                    # TODO: come up with own error class
                    raise AttributeError(
                        f"Experiment {name} has {experiment_num_classes} classes, not the expected {self.num_classes}!"
                    )

        self.experiments.update(temp_experiments)

    def add_metric(
        self,
        metric: str
        | typing.Type[Metric]
        | typing.Type[AveragedMetric]
        | typing.Iterable[str | typing.Type[Metric] | typing.Type[AveragedMetric]],
    ) -> None:
        self.metrics.add(metric)

    def add_experiment_aggregation(
        self, metric_name: str, aggregation_config: typing.Dict[str, typing.Any]
    ) -> None:
        aggregator = self.experiment_aggregations.get(aggregation_config, None)

        if aggregator is None:
            aggregator = get_experiment_aggregator(
                **aggregation_config,
                rng=self.rng,
            )

            self.experiment_aggregations[aggregation_config] = aggregator

        metric_instance = self.metrics[metric_name]
        self.metric_to_aggregator[metric_instance] = aggregator

    # TODO: document method
    def compute_metrics(
        self, sampling_method: str, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, typing.List[ExperimentResult]]:
        if len(self.metrics) == 0:
            raise ValueError(
                "No metrics have been added to the experiment yet. Use the `add_metric` method to add some."  # noqa: E501
            )

        if num_samples is None:
            num_samples = self.num_samples

        # Get the topological ordering of the metrics, such that no metric is computed before
        # its dependencies are
        metric_compute_order = self.metrics.get_compute_order()

        all_experiment_metric_values = defaultdict(list)
        for experiment_name, experiment in self.experiments.items():
            # TODO: parallelize metric computation???
            # First have the experiment generate synthetic confusion matrices and needed RootMetrics
            intermediate_stats: typing.Dict[
                RootMetric, ExperimentResult
            ] = experiment.sample(
                sampling_method=sampling_method, num_samples=num_samples
            )

            # Go through all metrics and dependencies in order
            for metric in metric_compute_order:
                # Root metrics have no dependency per-definition
                # and are already computed automatically
                if isinstance(metric, RootMetric):
                    continue

                # Filter out all the dependencies for the current metric
                # Since we allow each metric to define it's own dependencies by name (or alias)
                # We have to be a little lenient with how we look these up
                dependencies: typing.Dict[Metric, np.ndarray] = dict()
                for dependency_name in metric.dependencies:
                    dependency = metric_compute_order[dependency_name]
                    dependencies[dependency_name] = intermediate_stats[
                        dependency
                    ].values

                # Compute the current metric and add it to the dict
                metric_values = metric(**dependencies)

                # Add the metric values to the intermediate stats dictionary
                intermediate_stats[metric] = ExperimentResult(
                    experiment=self.experiments[experiment_name],
                    metric=metric,
                    values=metric_values,
                )

            # Filter out only the requested metrics
            # The keys are Metric instances, not str
            experiment_metric_values: typing.Dict[Metric, ExperimentResult] = {
                metric: intermediate_stats[metric] for metric in self.metrics
            }

            # Invert the key ordering
            for metric, metric_result in experiment_metric_values.items():
                all_experiment_metric_values[metric].append(metric_result)

        # Convert defaultdict to dict
        all_experiment_metric_values = {
            metric.name: experiment_results
            for metric, experiment_results in all_experiment_metric_values.items()
        }

        return all_experiment_metric_values

    def aggregate_experiments(
        self,
        metric_values: typing.Dict[str, typing.List[ExperimentResult]],
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes"]]:
        all_aggregated_vals = dict()

        # Iterate through the dict of metrics and values for each experiment
        for metric_name, experiment_results in metric_values.items():
            # TODO: implement check for assumption that all ExperimentResult objects use the same metric
            # and that this metric matches the metric name
            metric = experiment_results[0].metric
            # Fetch the right aggregator
            aggregator = self.metric_to_aggregator[metric]

            # Aggregate all the experiments into a summary distribution
            aggregation_result = aggregator(
                metric=metric, experiment_samples=experiment_results
            )

            # Wrap the output fromt he aggregator in a custom result dataclass
            aggregation_result = [
                ExperimentAggregationResult(
                    experiment_group=self,
                    aggregator=aggregator,
                    metric=metric,
                    heterogeneity=heterogeneity,
                    values=aggregated_vals,
                )
                for aggregated_vals, heterogeneity in aggregation_result
            ]

            # Record for output
            all_aggregated_vals[metric_name] = aggregation_result

        return all_aggregated_vals

    def __repr__(self) -> str:
        return f"ExperimentManager({self.name})"

    def __str__(self) -> str:
        return f"ExperimentManager({self.name})"


@dataclass(frozen=True)
class ExperimentAggregationResult:
    experiment_group: ExperimentManager
    aggregator: typing.Type[ExperimentAggregation]
    metric: typing.Type[Metric] | typing.Type[AveragedMetric]

    heterogeneity: HeterogeneityResult

    values: jtyping.Float[np.ndarray, " num_samples"]

    @property
    def name(self):
        return self.experiment_group.name
