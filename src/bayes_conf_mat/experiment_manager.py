import typing
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from warnings import warn

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
from bayes_conf_mat.experiment_aggregation.heterogeneity import (
    HeterogeneityResult,
)


# TODO: document class
class ExperimentManager:
    """_summary_

    Args:
        name (str): _description_
        experiments (typing.Dict[ str, typing.Dict  |  jtyping.Int[np.ndarray, 'num_classes num_classes'] ]): _description_
        num_samples (typing.Optional[int]): _description_
        seed (typing.Optional[int  |  np.random.BitGenerator]): _description_
        metrics (typing.Optional[typing.Iterable[str]], optional): _description_. Defaults to ().
        experiment_aggregations (typing.Optional[ typing.Dict[str, typing.Dict[str, str]] ], optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        num_samples: typing.Optional[int],
        seed: typing.Optional[int | np.random.BitGenerator],
        metrics: typing.Optional[typing.Iterable[str]] = (),
        experiment_aggregations: typing.Optional[
            typing.Dict[str, typing.Dict[str, str]]
        ] = None,
    ) -> None:
        self.name = name

        # Import hyperparameters
        # The number of synthetic confusion matrices to sample
        self.num_samples = num_samples

        # The manager's RNG
        if isinstance(seed, int) or isinstance(seed, float):
            self.rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.BitGenerator) or isinstance(
            seed, np.random.Generator
        ):
            self.rng = seed
        else:
            raise ValueError(
                f"Could not construct rng using seed of type `{type(seed)}`"
            )

        # The collection of experiments
        self.num_classes = None
        self.experiments = OrderedDict()

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

    def add_experiment(
        self,
        name: str,
        confusion_matrix: typing.Dict[str, typing.Any] | np.ndarray,
        prevalence_prior: float | str | jtyping.Float[np.ndarray, " num_classes"] = 0,
        confusion_prior: float | str | jtyping.Float[np.ndarray, " num_classes"] = 0,
    ):
        indep_rng = self.rng.spawn(1)[0]

        new_experiment = Experiment(
            # Provided for the user
            # Unique to each experiment
            name=name,
            confusion_matrix=confusion_matrix,
            prevalence_prior=prevalence_prior,
            confusion_prior=confusion_prior,
            # Provided by the manager
            rng=indep_rng,
        )

        if self.num_classes is None:
            self.num_classes = new_experiment.num_classes
        elif new_experiment.num_classes != self.num_classes:
            raise AttributeError(
                f"Experiment '{self.name}/{name}' has {new_experiment.num_classes} classes, not the expected {self.num_classes}"
            )

        if self.experiments.get(name, None) is not None:
            warn(f"Experiment '{self.name}/{name} alread exists. Overwriting.")

        self.experiments[name] = new_experiment

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
                rng=self.rng.spawn(1)[0],
            )

            self.experiment_aggregations[aggregation_config] = aggregator

        try:
            self.metrics[metric_name]
        except Exception as e:
            raise ValueError(
                f"Encountered error while trying to find metric {metric_name}: {e}"
            )

        self.metric_to_aggregator[metric_name] = aggregator

    # TODO: document method
    def compute_metrics(
        self, sampling_method: str, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[
        str,
        typing.Annotated[typing.List[ExperimentResult], " num_experiments num_classes"],
    ]:
        """
        _summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

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
            intermediate_stats: typing.Dict[RootMetric, ExperimentResult] = (
                experiment.sample(
                    sampling_method=sampling_method, num_samples=num_samples
                )
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
            experiment_metric_values: typing.Dict[
                Metric, typing.List[SplitExperimentResult]
            ] = {
                metric: experiment_result_to_split_experiment_results(
                    intermediate_stats[metric]
                )
                for metric in self.metrics
            }

            # Invert the key ordering
            for metric, metric_result in experiment_metric_values.items():
                all_experiment_metric_values[metric].append(metric_result)

        # Transpose the list of lists
        # i.e., go from [num_experiment, num_classes] to [num_classes, num_experiments]
        # Convert to dict
        all_experiment_metric_values: typing.Dict[
            Metric,
            typing.Annotated[
                typing.List[typing.List[ExperimentResult]],
                "num_classes num_experiments",
            ],
        ] = {
            metric.name: list(map(list, zip(*experiment_results)))
            for metric, experiment_results in all_experiment_metric_values.items()
        }

        return all_experiment_metric_values

    # TODO: document this method
    def aggregate_experiments(
        self,
        metric_values: typing.Dict[
            str,
            typing.Annotated[
                typing.List[typing.List[ExperimentResult]],
                "num_classes num_experiments",
            ],
        ],
    ) -> typing.Dict[str, typing.Annotated[typing.List, "num_classes"]]:
        all_aggregated_vals = dict()

        # Iterate through the dict of metrics and values for each experiment
        for metric_name, experiment_class_results in metric_values.items():
            # Fetch the right aggregator
            metric = self.metrics[metric_name]
            aggregator = self.metric_to_aggregator[metric_name]
            aggregation_results = []
            for experiment_results in experiment_class_results:
                # Aggregate all the experiments into a summary distribution
                aggregation_result = aggregator(
                    metric=metric, experiment_samples=experiment_results
                )

                aggregation_results.append(aggregation_result)

            # Wrap the output fromt he aggregator in a custom result dataclass
            aggregation_results = [
                ExperimentAggregationResult(
                    experiment_group=self,
                    aggregator=aggregator,
                    metric=metric,
                    class_index=i,
                    heterogeneity=heterogeneity,
                    values=aggregated_vals,
                )
                for i, (aggregated_vals, heterogeneity) in enumerate(
                    aggregation_results
                )
            ]

            # Record for output
            all_aggregated_vals[metric_name] = aggregation_results

        return all_aggregated_vals

    def __repr__(self) -> str:
        return f"ExperimentManager({self.name})"

    def __str__(self) -> str:
        return f"ExperimentManager({self.name})"


@dataclass(frozen=True)
class SplitExperimentResult(ExperimentResult):
    """Just like an experiment result, but now the values have been split across the individual classes.
    For convenience.

    Args:
        experiment (Experiment):
        metric (Metric | AveragedMetric):
        class_index (int): defaults to `None`.
        values: (Float[ndarray, "num_samples"])
    """

    experiment: Experiment
    metric: typing.Type[Metric] | typing.Type[AveragedMetric]
    values: jtyping.Float[np.ndarray, " num_samples"]
    class_index: typing.Optional[int] = None


def experiment_result_to_split_experiment_results(
    experiment_result: ExperimentResult,
) -> typing.List[SplitExperimentResult]:
    experiment_values = experiment_result.values
    metric = experiment_result.metric

    if metric.is_multiclass:
        split_experiment_results = [
            SplitExperimentResult(
                experiment=experiment_result.experiment,
                metric=experiment_result.metric,
                class_index=None,
                values=experiment_values,
            )
        ]

    else:
        split_experiment_values = np.split(
            ary=experiment_values,
            indices_or_sections=experiment_values.shape[1],
            axis=1,
        )

        split_experiment_results = [
            SplitExperimentResult(
                experiment=experiment_result.experiment,
                metric=experiment_result.metric,
                class_index=i,
                values=np.squeeze(vals),
            )
            for i, vals in enumerate(split_experiment_values)
        ]

    return split_experiment_results


@dataclass(frozen=True)
class ExperimentAggregationResult:
    experiment_group: ExperimentManager
    aggregator: typing.Type[ExperimentAggregation]
    metric: typing.Type[Metric] | typing.Type[AveragedMetric]
    heterogeneity: HeterogeneityResult
    values: jtyping.Float[np.ndarray, " num_samples"]
    class_index: typing.Optional[int] = None

    @property
    def name(self):
        return self.experiment_group.name
