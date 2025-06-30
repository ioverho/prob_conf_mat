from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import jaxtyping as jtyping

    from prob_conf_mat.metrics import MetricCollection
    from prob_conf_mat.experiment import ExperimentResult
    from prob_conf_mat.experiment_aggregation import ExperimentAggregator
    from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregationResult
    from prob_conf_mat.utils import RNG, MetricLike

from collections import OrderedDict
from warnings import warn
from dataclasses import dataclass

from prob_conf_mat.experiment import Experiment, ExperimentResult
from prob_conf_mat.experiment_aggregation import get_experiment_aggregator


@dataclass(frozen=True)
class ExperimentGroupResult:
    """A wrapper class for the output of an ExperimentGroup."""

    aggregation_result: OrderedDict[MetricLike, ExperimentAggregationResult]
    individual_experiment_results: OrderedDict[
        MetricLike,
        dict[Experiment, ExperimentResult],
    ]


class ExperimentGroup:
    """This class represents a group of related Experiments.

    The results across experiments can be aggregated to give an average result for the group.

    For example, this could represent the same model evaluated across different folds of the same
    dataset. Or they could be results on te same dataset from models with different
    weight initializations.

    Args:
        name (str): the name of this experiment group
        rng (RNG): the RNG used to control randomness
    """

    def __init__(
        self,
        name: str,
        rng: RNG,
    ) -> None:
        self.name = name

        # ======================================================================
        # Import hyperparameters
        # ======================================================================
        # The manager's RNG
        self.rng = rng

        # The collection of experiments
        self.num_classes = None
        self.experiments: OrderedDict = OrderedDict()

    @property
    def num_experiments(self) -> int:
        """The number of experiments in this ExperimentGroup."""
        return len(self.experiments)

    def __len__(self) -> int:
        """The number of experiments in this ExperimentGroup."""
        return self.num_experiments

    def add_experiment(
        self,
        name: str,
        confusion_matrix: jtyping.Int[np.ndarray, " num_classes num_classes"],
        prevalence_prior: str
        | float
        | jtyping.Float[np.typing.ArrayLike, " num_classes"] = 0,
        confusion_prior: str
        | float
        | jtyping.Float[np.typing.ArrayLike, " num_classes num_classes"] = 0,
    ) -> None:
        """Adds an Experiment to this ExperimentGroup.

        Each experiment is characterized by a single confusion matrix.

        Args:
            name (str): the name of this experiment
            confusion_matrix (jtyping.Int[np.ndarray, 'num_classes num_classes']): the
                confusion matrix for this experiment.
            prevalence_prior (str | float | Float[ArrayLike, ' num_classes'], optional): the prior
                over the prevalence counts for this experiments.
                Defaults to 0, Haldane's prior.
            confusion_prior (str | float | Float[ArrayLike, ' num_classes num_classes'], optional):
                the prior over the confusion counts for this experiments.
                Defaults to 0, Haldane's prior.

        """
        # Spawn a new RNG, that is independent of the RNGs used in other experiments of this
        # experiment group
        indep_rng = self.rng.spawn(1)[0]

        # Define the experiment
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

        # Check to make sure that the number of classes in the new experiment matches the number of
        # classes in all other experiment
        # This really is the minimal requirement for an experiment group
        if self.num_classes is None:
            self.num_classes = new_experiment.num_classes
        elif new_experiment.num_classes != self.num_classes:
            raise AttributeError(
                (
                    f"Experiment '{self.name}/{name}' has {new_experiment.num_classes} classes, "
                    f"not the expected {self.num_classes}"
                ),
            )

        # Check if this experiment already exists
        # Overwrite if so
        if self.experiments.get(name, None) is not None:  # type: ignore
            warn(
                message=f"Experiment '{self.name}/{name}' already exists. Overwriting.",
            )

        # Finally, add the experiment to the experiment store
        self.experiments[name] = new_experiment

    def __getitem__(self, key: str) -> Experiment:
        """Gets an experiment under this ExperimentGroup by its name.

        Args:
            key (str): the experiment name

        Returns:
            Experiment: _description_
        """
        if key not in self.experiments:
            raise KeyError(
                (
                    f"No experiment with name '{key}' is currently present in experiment group "
                    f"'{self.name}'."
                ),
            )

        return self.experiments[key]

    def sample_metrics(  # noqa: D102
        self,
        metrics: MetricCollection,
        sampling_method: str,
        num_samples: int,
        metric_to_aggregator: dict[MetricLike, ExperimentAggregator],
    ) -> ExperimentGroupResult:
        """Samples and aggregates metrics from Experiments belonging to this ExperimentGroup.

        Args:
            metrics (MetricCollection): the collection of metrics for which samples are needed.
            sampling_method (str): the sampling method used to construct
                synthetic confusion matrices.
            num_samples (int): the number of synthetic confusion matrices to sample.
            metric_to_aggregator (_type_): how each metric should be aggregated.

        Returns:
            ExperimentGroupResult: the output of all Experiments and their aggregation
        """
        # Compute metrics for each experiment and store them
        all_metrics_experiment_results: dict[
            MetricLike,
            list[ExperimentResult],
        ] = {metric: [] for metric in metrics.get_insert_order()}
        for _, experiment in self.experiments.items():
            all_metrics_experiment_result: dict[MetricLike, ExperimentResult] = (
                experiment.sample_metrics(
                    metrics=metrics,
                    sampling_method=sampling_method,
                    num_samples=num_samples,
                )
            )

            for metric, experiment_result in all_metrics_experiment_result.items():
                all_metrics_experiment_results[metric].append(experiment_result)

        # Iterate over all the individual experiments and aggregate them
        all_metrics_experiment_aggregation_result = dict()
        for metric, experiment_results in all_metrics_experiment_results.items():
            # Fetch the experiment aggregation method
            aggregator: ExperimentAggregator | None = metric_to_aggregator.get(
                metric,
                None,
            )

            # Handle missing experiment aggregation methods
            if aggregator is None:
                if len(self) > 1:
                    raise ValueError(
                        (
                            f"Metric '{metric.name}' does not have an assigned aggregation method, "
                            f"but experiment group '{self.name}' has {len(self)} experiments. "
                            f"Try adding one using "
                            f"`Study.add_metric(metric={metric.name}, aggregation=...)`."
                        ),
                    )

                # We're allowed to pass the group's RNG, because the singleton aggregator is just
                # an identity function
                aggregator = get_experiment_aggregator(
                    aggregation="singleton",
                    rng=self.rng,
                )

            # Run the aggregation
            experiment_aggregation_result: dict[
                MetricLike,
                ExperimentAggregationResult,
            ] = aggregator(
                experiment_group=self,
                metric=metric,
                experiment_results=experiment_results,
            )  # type: ignore

            all_metrics_experiment_aggregation_result[metric] = (
                experiment_aggregation_result
            )

        # Clean the output
        # All metrics in insertion order
        # Have a nested dict for the experiment results
        all_metrics_experiment_results_cleaned: OrderedDict[
            MetricLike,
            dict[Experiment, ExperimentResult],
        ] = OrderedDict(
            [
                (
                    metric,
                    {
                        experiment_result.experiment: experiment_result
                        for experiment_result in all_metrics_experiment_results[metric]
                    },
                )
                for metric in metrics.get_insert_order()
            ],
        )

        all_metrics_experiment_aggregation_result_cleaned: OrderedDict[
            MetricLike,
            ExperimentAggregationResult,
        ] = OrderedDict(
            [
                (metric, all_metrics_experiment_aggregation_result[metric])
                for metric in metrics.get_insert_order()
            ],
        )

        return ExperimentGroupResult(
            aggregation_result=all_metrics_experiment_aggregation_result_cleaned,
            individual_experiment_results=all_metrics_experiment_results_cleaned,
        )

    def __repr__(self) -> str:  # noqa: D105
        return f"ExperimentGroup({self.name})"

    def __str__(self) -> str:  # noqa: D105
        return f"ExperimentGroup({self.name})"
