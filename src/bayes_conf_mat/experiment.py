from __future__ import annotations

import typing
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.io import get_io
from bayes_conf_mat.metrics import RootMetric, Metric, AveragedMetric, MetricCollection
from bayes_conf_mat.stats import dirichlet_sample, dirichlet_prior


class SamplingMethod(Enum):
    POSTERIOR = "posterior"
    PRIOR = "prior"
    RANDOM = "random"
    INPUT = "input"


@dataclass(frozen=True)
class ExperimentResult:
    """Storage for results from an experiment

    Args:
        experiment (Experiment): the experiment which produced these results
        metric (typing.Type[Metric | AveragedMetric]): the metric instance that produced these results
        values (jtyping.Float[np.ndarray, " num_samples #num_classes"]): the actual produced values
    """

    experiment: Experiment
    metric: typing.Type[Metric | AveragedMetric]

    values: jtyping.Float[np.ndarray, " num_samples #num_classes"]

    @property
    def is_multiclass(self):
        return self.metric.is_multiclass

    @property
    def bounds(self):
        return self.metric.bounds

    @property
    def num_classes(self):
        return self.experiment.num_classes

    @property
    def num_samples(self):
        return self.values.shape[0]

    def __repr__(self):
        return f"ExperimentResult(experiment={self.experiment}, metric={self.metric})"

    def __str__(self):
        return f"ExperimentResult(experiment={self.experiment}, metric={self.metric})"


# TODO: document this class
class Experiment:
    def __init__(
        self,
        name: str,
        rng: np.random.BitGenerator,
        confusion_matrix: typing.Dict
        | jtyping.Float[np.typing.ArrayLike, " num_classes num_classes"],
        prevalence_prior: str | int | jtyping.Int[np.ndarray, " num_classes"],
        confusion_prior: str
        | int
        | jtyping.Int[np.ndarray, " num_classes num_classes"],
    ) -> None:
        self.name = name

        # Argument Validation ==================================================
        # Import the confusion matrix
        # Check if config like object
        if hasattr(confusion_matrix, "items"):
            self.confusion_matrix_loader = get_io(**confusion_matrix)

        # If a numpy array, just store it
        elif isinstance(confusion_matrix, np.ndarray):
            self.confusion_matrix_loader = get_io(
                format="in_memory", data=confusion_matrix
            )

        # If not a numpy array, tries to make it one
        else:
            try:
                confusion_matrix = np.array(confusion_matrix)
                self.confusion_matrix_loader = get_io(
                    format="in_memory", data=confusion_matrix
                )
            except Exception as e:
                raise ValueError(
                    f"Ran into exception when trying to convert {confusion_matrix} into a np.ndarray:\n{e}"
                )

        # load and validate the confusion matrix
        self.confusion_matrix = self.confusion_matrix_loader.load()

        # The prior strategy used for defining the Dirichlet prior counts
        self._init_prevalence_prior = prevalence_prior
        self.prevalence_prior = dirichlet_prior(
            prevalence_prior, shape=(self.num_classes,)
        )

        self._init_confusion_prior = confusion_prior
        self.confusion_prior = dirichlet_prior(
            confusion_prior, shape=(self.num_classes, self.num_classes)
        )

        # The RNG
        self.rng = rng

    @property
    def num_classes(self):
        return self.confusion_matrix.shape[0]

    @property
    def num_predictions(self):
        return np.sum(self.confusion_matrix)

    def _wrap_sample_result(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
        p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_pred_given_condition: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
        p_pred: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_condition_given_pred: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ):
        experiment_sample_result = {
            RootMetric("norm_confusion_matrix"): ExperimentResult(
                experiment=self,
                metric=RootMetric("norm_confusion_matrix"),
                values=norm_confusion_matrix,
            ),
            RootMetric("p_condition"): ExperimentResult(
                experiment=self,
                metric=RootMetric("p_condition"),
                values=p_condition,
            ),
            RootMetric("p_pred_given_condition"): ExperimentResult(
                experiment=self,
                metric=RootMetric("p_pred_given_condition"),
                values=p_pred_given_condition,
            ),
            RootMetric("p_pred"): ExperimentResult(
                experiment=self,
                metric=RootMetric("p_pred"),
                values=p_pred,
            ),
            RootMetric("p_condition_given_pred"): ExperimentResult(
                experiment=self,
                metric=RootMetric("p_condition_given_pred"),
                values=p_condition_given_pred,
            ),
        }

        return experiment_sample_result

    def _sample(
        self,
        condition_counts: jtyping.Float[np.ndarray, " num_classes"],
        confusion_matrix: jtyping.Float[np.ndarray, " num_classes num_classes"],
        num_samples: typing.Optional[int] = None,
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        p_condition = dirichlet_sample(
            rng=self.rng,
            alphas=condition_counts,
            num_samples=num_samples,
        )

        p_pred_given_condition = dirichlet_sample(
            rng=self.rng,
            alphas=confusion_matrix,
            num_samples=num_samples,
        )

        norm_confusion_matrix = p_pred_given_condition * p_condition[:, :, np.newaxis]

        p_pred = norm_confusion_matrix.sum(axis=1)

        p_condition_given_pred = norm_confusion_matrix / p_pred[:, np.newaxis, :]

        output = self._wrap_sample_result(
            norm_confusion_matrix=norm_confusion_matrix,
            p_condition=p_condition,
            p_pred_given_condition=p_pred_given_condition,
            p_pred=p_pred,
            p_condition_given_pred=p_condition_given_pred,
        )

        return output

    def sample_input(
        self,
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        """For debug purposes: uses the input confusion matrix as the samples.

        Essentially just adds a batch dimension to the existing confusion matrix.
        """
        confusion_matrix = self.confusion_matrix[np.newaxis, :, :]

        norm_confusion_matrix = confusion_matrix / confusion_matrix.sum()

        condition_counts = confusion_matrix.sum(axis=2)
        p_condition = condition_counts / condition_counts.sum()

        p_pred_given_condition = norm_confusion_matrix / p_condition[:, np.newaxis, :]

        prediction_counts = confusion_matrix.sum(axis=1)
        p_pred = prediction_counts / prediction_counts.sum()

        p_condition_given_pred = norm_confusion_matrix / p_pred[:, :, np.newaxis]

        output = self._wrap_sample_result(
            norm_confusion_matrix=norm_confusion_matrix,
            p_condition=p_condition,
            p_pred_given_condition=p_pred_given_condition,
            p_pred=p_pred,
            p_condition_given_pred=p_condition_given_pred,
        )

        return output

    def sample_prior(
        self, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        return self._sample(
            num_samples=num_samples,
            condition_counts=self.prevalence_prior,
            confusion_matrix=self.confusion_prior,
        )

    def sample_posterior(
        self, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        condition_counts = self.confusion_matrix.sum(axis=1)
        posterior_condition_counts = self.prevalence_prior + condition_counts

        posterior_pred_given_condtion_counts = (
            self.confusion_prior + self.confusion_matrix
        )

        return self._sample(
            num_samples=num_samples,
            condition_counts=posterior_condition_counts,
            confusion_matrix=posterior_pred_given_condtion_counts,
        )

    def sample_random_model(
        self, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        """Sample from the randomly initialized model distribution.
        It uses the class prevalence from the data, but a random confusion matrix.
        Thus, this should model a random classifier on the used dataset, accounting for class imbalance.

        Args:
            num_samples (typing.Optional[int], optional): _description_. Defaults to None.

        Returns:
            typing.Dict[str, np.ndarray]: _description_
        """  # noqa: E501

        condition_counts = self.confusion_matrix.sum(axis=1)
        posterior_condition_counts = self.prevalence_prior + condition_counts

        posterior_pred_given_condtion_counts = (
            self.confusion_prior + self.confusion_matrix
        )

        # Averages over the rows
        random_pred_given_condition_counts = np.broadcast_to(
            np.mean(
                posterior_pred_given_condtion_counts,
                axis=-1,
                keepdims=True,
            ),
            (self.num_classes, self.num_classes),
        )

        return self._sample(
            num_samples=num_samples,
            condition_counts=posterior_condition_counts,
            confusion_matrix=random_pred_given_condition_counts,
        )

    # TODO: document this method
    def sample(
        self,
        sampling_method: SamplingMethod,
        num_samples: int,
    ) -> typing.Dict[RootMetric, ExperimentResult]:
        """Sample synthetic confusion matrices corresponding to this experiment.

        Args:
            sampling_method (SamplingMethod): the sampling method used to generate the metric values. Must a member of the SamplingMethod enum
            num_samples (int): the number of synthetic confusion matrices to sample

        Returns:
            typing.Dict[RootMetric, ExperimentResult]: a dictionary of RootMetric instances
        """
        root_metrics = dict()

        match sampling_method:
            case SamplingMethod.POSTERIOR.value:
                root_metrics.update(self.sample_posterior(num_samples=num_samples))
            case SamplingMethod.PRIOR.value:
                root_metrics.update(self.sample_prior(num_samples=num_samples))
            case SamplingMethod.RANDOM.value:
                root_metrics.update(self.sample_random_model(num_samples=num_samples))
            case SamplingMethod.INPUT.value:
                root_metrics.update(self.sample_input())
            case _:
                raise ValueError(
                    f"Parameter `sampling_method` must be one of {tuple(sm.value for sm in SamplingMethod)}. Currently: {sampling_method}"
                )

        return root_metrics

    def sample_metrics(
        self,
        metrics: MetricCollection,
        sampling_method: SamplingMethod,
        num_samples: int,
    ) -> typing.Dict[RootMetric | Metric | AveragedMetric, ExperimentResult]:
        """_summary_

        Args:
            metrics (MetricCollection): the metrics needed to be computed on the synthetic confusion matrices
            sampling_method (SamplingMethod): the sampling method used to generate the metric values. Must a member of the SamplingMethod enum
            num_samples (int): the number of synthetic confusion matrices to sample

        Returns:
            typing.Mapping[Metric | AveragedMetric, ExperimentResult]: a mapping from metric instance to an `ExperimentResult` instance
        """

        # Get the topological ordering of the metrics, such that no metric is computed before
        # its dependencies are
        metric_compute_order = metrics.get_compute_order()

        # First have the experiment generate synthetic confusion matrices and needed RootMetrics
        # typing.Dict[RootMetric, ExperimentResult]
        intermediate_results = self.sample(
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

                dependencies[dependency_name] = intermediate_results[dependency].values

            # Compute the current metric and add it to the dict
            metric_values = metric(**dependencies)

            # Add the metric values to the intermediate stats dictionary
            intermediate_results[metric] = ExperimentResult(
                experiment=self,
                metric=metric,
                values=metric_values,
            )

        results = OrderedDict(
            [
                (metric, intermediate_results[metric])
                for metric in metrics.get_insert_order()
            ]
        )

        return results

    def __repr__(self) -> str:
        return f"Experiment({self.name})"

    def __str__(self) -> str:
        return f"Experiment({self.name})"
