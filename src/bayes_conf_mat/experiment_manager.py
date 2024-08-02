from __future__ import annotations

import typing
from collections import OrderedDict, namedtuple
from warnings import warn

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment import Experiment, ExperimentResult
from bayes_conf_mat.metrics import MetricCollection
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator


output_wrapper = namedtuple(
    "ExperimentManagerOutput",
    field_names=[
        "aggregation_result",
        "individual_experiment_results",
    ],
)


# TODO: document class
class ExperimentManager:
    def __init__(
        self,
        name: str,
        seed: typing.Optional[int | np.random.BitGenerator],
    ) -> None:
        self.name = name

        # ======================================================================
        # Import hyperparameters
        # ======================================================================
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

    @property
    def num_experiments(self):
        return len(self.experiments)

    def __len__(self):
        return self.num_experiments

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

    # TODO: document method
    def sample_metrics(
        self,
        metrics: MetricCollection,
        sampling_method: str,
        num_samples: int,
        metric_to_aggregator,
    ) -> typing.Dict[
        str,
        typing.Annotated[
            typing.List["ExperimentResult"], " num_experiments num_classes"
        ],
    ]:
        # Compute metrics for each experiment and store them
        all_metrics_experiment_results = {
            metric: [] for metric in metrics.get_insert_order()
        }
        for _, experiment in self.experiments.items():
            all_metrics_experiment_result = experiment.sample_metrics(
                metrics=metrics,
                sampling_method=sampling_method,
                num_samples=num_samples,
            )

            for metric, experiment_result in all_metrics_experiment_result.items():
                all_metrics_experiment_results[metric].append(experiment_result)

        # Iterate over all the individual experiments and aggregate them
        all_metrics_experiment_aggregation_result = dict()
        for metric, experiment_results in all_metrics_experiment_results.items():
            # Fetch the experiment aggregation method
            aggregator = metric_to_aggregator.get(metric, None)

            # Handle missing experiment aggregation methods
            if aggregator is None:
                if len(self) > 1:
                    raise ValueError(
                        f"Metric '{metric}' does not have an assigned aggregation method, but experiment group {self} has {len(self)} experiments. Try adding one using `Study.add_experiment_aggregation`."
                    )
                else:
                    aggregator = get_experiment_aggregator(
                        aggregation="singleton", rng=None
                    )

            # Run the aggregation
            experiment_aggregation_result = aggregator(
                experiment_group=self,
                metric=metric,
                experiment_results=experiment_results,
            )

            all_metrics_experiment_aggregation_result[metric] = (
                experiment_aggregation_result
            )

        # Clean the output
        # All metrics in insertion order
        # Have a nested dict for the experiment results
        all_metrics_experiment_results = OrderedDict(
            [
                (
                    metric,
                    {
                        experiment_result.experiment: experiment_result
                        for experiment_result in all_metrics_experiment_results[metric]
                    },
                )
                for metric in metrics.get_insert_order()
            ]
        )

        all_metrics_experiment_aggregation_result = OrderedDict(
            [
                (metric, all_metrics_experiment_aggregation_result[metric])
                for metric in metrics.get_insert_order()
            ]
        )

        return output_wrapper(
            aggregation_result=all_metrics_experiment_aggregation_result,
            individual_experiment_results=all_metrics_experiment_results,
        )

    def __repr__(self) -> str:
        return f"ExperimentManager({self.name})"

    def __str__(self) -> str:
        return f"ExperimentManager({self.name})"
