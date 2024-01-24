import typing
from collections import defaultdict, OrderedDict

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics.base import Metric, AggregatedMetric, RootMetric
from bayes_conf_mat.metrics.collection import MetricCollection
from bayes_conf_mat.experiment import Experiment
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator


class ExperimentManager:
    def __init__(
        self,
        experiments: typing.Dict[
            str, jtyping.Int[np.ndarray, " num_classes num_classes"]
        ],
        num_samples: typing.Optional[int] = None,
        seed: typing.Optional[int | np.random.BitGenerator] = 0,
        num_proc: typing.Optional[int] = 0,
        prior_strategy: str = "laplace",
        metrics: typing.Optional[typing.Iterable[str]] = (),
        experiment_aggregations: typing.Optional[
            typing.Dict[str, typing.Dict[str, str]]
        ] = None,
    ) -> None:
        # Import hyperparameters
        # The number of synthetic confusion matrices to sample
        self.num_samples = num_samples

        # The prior strategy to use for each experiment
        self.prior_strategy = prior_strategy

        # The RNG
        if isinstance(seed, int) or isinstance(seed, float):
            self.rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.BitGenerator) or isinstance(
            seed, np.random.Generator
        ):
            self.rng = seed

        # The number of processes to use for parallelization
        self.num_proc = num_proc

        # Collection of the experiments
        self.experiments = OrderedDict()
        self.add_experiments(experiments)

        # Collection of the metrics used
        self.metrics = MetricCollection(metrics)

        # Collection of experiment aggregators
        self.experiment_aggregations = dict()
        self.metric_to_aggregator = dict()
        if experiment_aggregations is not None:
            for metric_name, aggregation_config in experiment_aggregations.items():
                self.add_experiment_aggregation(
                    metric_name=metric_name, aggregation_config=aggregation_config
                )

    def add_experiments(
        self,
        experiments: typing.Dict[
            str, jtyping.Int[np.ndarray, " num_classes num_classes"]
        ],
    ) -> None:
        temp_experiments = OrderedDict()
        for name, confusion_matrix in experiments.items():
            temp_experiments[name] = Experiment(
                name=name,
                confusion_matrix=confusion_matrix,
                rng=self.rng,
                prior_strategy=self.prior_strategy,
            )

        self.experiments.update(temp_experiments)

    def add_metric(
        self,
        metric: str
        | typing.Type[Metric]
        | typing.Type[AggregatedMetric]
        | typing.Iterable[str | typing.Type[Metric] | typing.Type[AggregatedMetric]],
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
                num_proc=self.num_proc,
            )

            self.experiment_aggregations[aggregation_config] = aggregator

        metric_instance = self.metrics[metric_name]
        self.metric_to_aggregator[metric_instance] = aggregator

    def compute_metrics(
        self, sampling_method: str, num_samples: typing.Optional[int] = None
    ):
        if len(self.metrics) == 0:
            raise ValueError(
                "No metrics have been added to the experiment yet. Use the `add_metric` method to add some."  # noqa: E501
            )

        if num_samples is None:
            num_samples = self.num_samples

        metric_compute_order = self.metrics.get_compute_order()

        all_experiment_metric_values = defaultdict(OrderedDict)
        for experiment_name, experiment in self.experiments.items():
            # First have the experiment generate synthetic confusion matrices and needed root metrics
            intermediate_stats = experiment.sample(
                sampling_method=sampling_method, num_samples=num_samples
            )

            # Go through all metrics and dependencies in order
            for metric in metric_compute_order:
                # Root metrics have no dependency per-definition
                # and are already computed automatically
                if isinstance(metric, RootMetric):
                    continue

                # Filter out all the dependencies for the current metric
                dependencies = dict()
                for dependency_name in metric.dependencies:
                    # Get the dependency's name
                    # Unfortunately this means we have to instantiate another metric
                    # But that's the cost of allowing any dependency
                    # Should get garbage collected out immediately
                    dependency = metric_compute_order[dependency_name]
                    dependencies[dependency_name] = intermediate_stats[dependency]

                # Compute the current metric and add it to the dict
                values = metric(**dependencies)

                # Add the metric values to the intermediate stats dictionary
                intermediate_stats[metric] = values

            # Filter out only the requested metrics
            # The keys are Metric instances, not str
            experiment_metric_values = {
                metric: intermediate_stats[metric] for metric in self.metrics
            }

            # Invert the key ordering
            for metric_name, metric_values in experiment_metric_values.items():
                all_experiment_metric_values[metric_name][
                    experiment_name
                ] = metric_values

        # Convert defaultdict to dict
        all_experiment_metric_values = dict(all_experiment_metric_values)

        return all_experiment_metric_values

    def aggregate_experiments(
        self,
        metric_values: typing.Dict[
            Metric | AggregatedMetric,
            typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes"]],
        ],
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes"]]:
        all_aggregated_vals = OrderedDict()

        # Iterate through the dict of metrics and values for each experiment
        for metric, metric_vals in metric_values.items():
            # Fetch the right aggregator
            aggregator = self.metric_to_aggregator[metric]

            # Aggregate all the experiments into a summary distribution
            aggregated_vals = aggregator(experiment_samples=metric_vals, metric=metric)

            # Record for output
            all_aggregated_vals[metric] = aggregated_vals

        return all_aggregated_vals
