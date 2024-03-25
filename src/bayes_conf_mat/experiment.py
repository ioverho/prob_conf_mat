import typing
from dataclasses import dataclass

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.io import get_io
from bayes_conf_mat.metrics import RootMetric, Metric, AveragedMetric
from bayes_conf_mat.stats import dirichlet_sample, dirichlet_prior

_IMPLEMENTED_SAMPLING_METHODS = {"prior", "posterior", "random", "input"}


# TODO: document this class
class Experiment:
    def __init__(
        self,
        name: str,
        confusion_matrix: typing.Dict
        | jtyping.Int[np.ndarray, " num_classes num_classes"],
        rng: np.random.BitGenerator,
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
            confusion_matrix_loader = get_io(**confusion_matrix)
            self.confusion_matrix = confusion_matrix_loader.load()

        # If a numpy array, just store it
        elif isinstance(confusion_matrix, np.ndarray):
            self.confusion_matrix = confusion_matrix

        # If not a numpy array, tries to make it one
        else:
            try:
                self.confusion_matrix = np.array(confusion_matrix)
            except Exception as e:
                raise ValueError(
                    f"Ran into exception when trying to convert {confusion_matrix} into a np.ndarray:\n{e}"
                )

        # The prior strategy used for defining the Dirichlet prior counts
        self.prevalence_prior = dirichlet_prior(
            prevalence_prior, shape=(self.num_classes,)
        )

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
        sampling_method: str,
        num_samples: int,
    ):
        root_metrics = dict()

        if sampling_method not in _IMPLEMENTED_SAMPLING_METHODS:
            raise ValueError(
                f"Sampling method must be one of `{_IMPLEMENTED_SAMPLING_METHODS}`"
            )  # noqa: E501

        elif sampling_method == "posterior":
            root_metrics.update(self.sample_posterior(num_samples=num_samples))

        elif sampling_method == "prior":
            root_metrics.update(self.sample_prior(num_samples=num_samples))

        elif sampling_method == "posterior":
            root_metrics.update(self.sample_random_model(num_samples=num_samples))

        elif sampling_method == "input":
            root_metrics.update(self.sample_input())

        return root_metrics

    def __repr__(self) -> str:
        return f"Experiment({self.name})"

    def __str__(self) -> str:
        return f"Experiment({self.name})"


@dataclass(frozen=True)
class ExperimentResult:
    experiment: Experiment
    metric: typing.Type[Metric] | typing.Type[AveragedMetric]

    values: jtyping.Float[np.ndarray, " num_samples *num_classes"]

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
