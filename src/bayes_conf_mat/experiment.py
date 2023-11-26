import typing
import warnings
from collections.abc import Iterable

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.metrics.base import RootMetric, Metric, AggregatedMetric
from bayes_conf_mat.math.dirichlet_distribution import dirichlet_sample, dirichlet_prior
from bayes_conf_mat.utils._scheduling import generate_metric_computation_schedule

_IMPLEMENTED_SAMPLING_METHODS = {"prior", "posterior", "random", "input"}


class Experiment:
    def __init__(
        self,
        confusion_matrix: jtyping.Int[np.ndarray, " num_classes num_classes"],
        num_samples: typing.Optional[int] = None,
        seed: int = 0,
        prior_strategy: str = "laplace",
        metrics: typing.Optional[typing.Iterable[str]] = (),
    ) -> None:
        # Argument Validation ==================================================
        # Import the confusion matrix
        # If not a numpy array, tries to make it one
        self.confusion_matrix = confusion_matrix
        if not isinstance(self.confusion_matrix, np.ndarray):
            self.confusion_matrix = np.array(self.confusion_matrix)

        # The prior strategy used for defining the Dirichlet prior counts
        self.prior_strategy = prior_strategy

        # The number of synthetic confusion matrices to sample
        self.num_samples = num_samples

        # The RNG
        if isinstance(seed, int) or isinstance(seed, float):
            self._rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.BitGenerator) or isinstance(
            seed, np.random.Generator
        ):
            self._rng = seed

        # Other Stuff ==========================================================
        self._metric_names = list()
        self._metrics = list()
        self._metrics_set = set()
        self.add_metric(metric=metrics)

    @property
    def num_classes(self):
        return self.confusion_matrix.shape[0]

    @property
    def num_predictions(self):
        return np.sum(self.confusion_matrix)

    @property
    def N(self):
        return self.num_predictions

    def _sample(
        self,
        condition_counts: jtyping.Float[np.ndarray, " num_classes"],
        confusion_matrix: jtyping.Float[np.ndarray, " num_classes num_classes"],
        num_samples: typing.Optional[int] = None,
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        if self.num_samples is None and num_samples is None:
            raise ValueError("Must specify `num_samples`")
        elif num_samples is None:
            num_samples = self.num_samples

        p_condition = dirichlet_sample(
            rng=self._rng,
            alphas=condition_counts,
            num_samples=num_samples,
        )

        p_pred_given_condition = dirichlet_sample(
            rng=self._rng,
            alphas=confusion_matrix,
            num_samples=num_samples,
        )

        norm_confusion_matrix = p_pred_given_condition * p_condition[:, :, np.newaxis]

        p_pred = norm_confusion_matrix.sum(axis=1)

        p_condition_given_pred = norm_confusion_matrix / p_pred[:, np.newaxis, :]

        output_dict = {
            "norm_confusion_matrix": norm_confusion_matrix,
            "p_condition": p_condition,
            "p_pred_given_condition": p_pred_given_condition,
            "p_pred": p_pred,
            "p_condition_given_pred": p_condition_given_pred,
        }

        return output_dict

    def sample_prior(
        self, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        prior_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes,)
        )
        prior_pred_given_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes, self.num_classes)
        )

        return self._sample(
            num_samples=num_samples,
            condition_counts=prior_condition_counts,
            confusion_matrix=prior_pred_given_condition_counts,
        )

    def sample_posterior(
        self, num_samples: typing.Optional[int] = None
    ) -> typing.Dict[str, jtyping.Float[np.ndarray, " num_samples num_classes ..."]]:
        prior_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes,)
        )
        prior_pred_given_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes, self.num_classes)
        )

        condition_counts = self.confusion_matrix.sum(axis=1)

        posterior_condition_counts = prior_condition_counts + condition_counts
        posterior_pred_given_condtion_counts = (
            prior_pred_given_condition_counts + self.confusion_matrix
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
        prior_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes,)
        )
        prior_pred_given_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes, self.num_classes)
        )

        condition_counts = self.confusion_matrix.sum(axis=1)

        posterior_condition_counts = prior_condition_counts + condition_counts
        posterior_pred_given_condtion_counts = (
            prior_pred_given_condition_counts + self.confusion_matrix
        )

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

        output_dict = {
            "norm_confusion_matrix": norm_confusion_matrix,
            "p_condition": p_condition,
            "p_pred_given_condition": p_pred_given_condition,
            "p_pred": p_pred,
            "p_condition_given_pred": p_condition_given_pred,
        }

        return output_dict

    def add_metric(
        self,
        metric: str
        | typing.Type[Metric]
        | typing.Type[AggregatedMetric]
        | typing.Iterable[str | typing.Type[Metric] | typing.Type[AggregatedMetric]],
    ):
        """_summary_

        Args:
            metric (str | typing.Type[Metric] | typing.Type[AggregatedMetric] | typing.Iterable[str  |  typing.Type[Metric]  |  typing.Type[AggregatedMetric]]): _description_
        """  # noqa: E501
        if isinstance(metric, Iterable):
            for m in metric:
                self._add_metric(m)
        else:
            self._add_metric(metric)

    def _add_metric(self, metric: str | Metric | AggregatedMetric):
        if isinstance(metric, str):
            metric_instance = get_metric(metric)

            if metric_instance in self._metrics_set:
                warnings.warn(
                    f"Metric `{metric}` already added to experiment. Skipping."
                )  # noqa: E501
                return None

            self._metric_names.append(metric)
            # TODO: replace with ordered dict
            self._metrics.append(metric_instance)
            self._metrics_set.add(metric_instance)

        elif issubclass(metric.__class__, Metric) or issubclass(
            metric.__class__, AggregatedMetric
        ):
            self._metric_names.append(metric.name)
            self._metrics.append(metric)
            self._metrics_set.add(metric)

        else:
            raise TypeError(
                f"Metric must be of type `str`, or a subclass of `Metric` or `AggregatedMetric`, not {metric}: {type(metric)}"  # noqa: E501
            )

    def _compute_metrics(
        self, sample_method: str, num_samples: typing.Optional[int] = None
    ):
        if len(self._metrics) == 0:
            raise ValueError(
                "No metrics have been added to the experiment yet. Use the `add_metric` method to add some."  # noqa: E501
            )

        intermediate_stats = dict()

        if sample_method not in _IMPLEMENTED_SAMPLING_METHODS:
            raise ValueError(
                f"Sampling method must be one of `{_IMPLEMENTED_SAMPLING_METHODS}`"
            )  # noqa: E501

        elif sample_method == "posterior":
            intermediate_stats.update(self.sample_posterior(num_samples=num_samples))

        elif sample_method == "prior":
            intermediate_stats.update(self.sample_prior(num_samples=num_samples))

        elif sample_method == "posterior":
            intermediate_stats.update(self.sample_random_model(num_samples=num_samples))

        elif sample_method == "input":
            intermediate_stats.update(self.sample_input())

        computation_schedule = generate_metric_computation_schedule(self._metrics)

        for metric in computation_schedule:
            if isinstance(metric, RootMetric):
                continue

            dependencies = {
                dependency: intermediate_stats[get_metric(dependency).name]
                for dependency in metric.dependencies
            }

            values = metric(**dependencies)

            intermediate_stats[metric.name] = values

        reported_metrics = {
            self._metric_names[i]: intermediate_stats[metric.name]
            for i, metric in enumerate(self._metrics)
        }

        return reported_metrics
