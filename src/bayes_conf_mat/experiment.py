import typing
import datetime

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.metrics.base import RootMetric
from bayes_conf_mat.math.dirichlet_distribution import dirichlet_sample, dirichlet_prior

_IMPLEMENTED_SAMPLING_METHODS = {"prior", "posterior", "random", "input"}


class Experiment:
    def __init__(
        self,
        name: str,
        confusion_matrix: jtyping.Int[np.ndarray, " num_classes num_classes"],
        rng: np.random.BitGenerator,
        prior_strategy: str = "laplace",
    ) -> None:
        self.time_stamp = format(datetime.datetime.now(), "%y%m%d-%H:%M:%S")
        self.name = name

        # Argument Validation ==================================================
        # Import the confusion matrix
        # If not a numpy array, tries to make it one
        self.confusion_matrix = confusion_matrix
        if not isinstance(self.confusion_matrix, np.ndarray):
            self.confusion_matrix = np.array(self.confusion_matrix)

        # The prior strategy used for defining the Dirichlet prior counts
        self.prior_strategy = prior_strategy

        # The RNG
        self.rng = rng

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

        output_dict = {
            RootMetric("norm_confusion_matrix"): norm_confusion_matrix,
            RootMetric("p_condition"): p_condition,
            RootMetric("p_pred_given_condition"): p_pred_given_condition,
            RootMetric("p_pred"): p_pred,
            RootMetric("p_condition_given_pred"): p_condition_given_pred,
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

    def sample(
        self,
        sampling_method: str,
        num_samples: int,
    ):
        # if len(self.metrics) == 0:
        #    raise ValueError(
        #        "No metrics have been added to the experiment yet. Use the `add_metric` method to add some."  # noqa: E501
        #    )

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

    # TODO: move this method to experiment manager class
    def compute_metrics(
        self, sample_method: str, num_samples: typing.Optional[int] = None
    ):
        if len(self.metrics) == 0:
            raise ValueError(
                "No metrics have been added to the experiment yet. Use the `add_metric` method to add some."  # noqa: E501
            )

        if num_samples is None:
            num_samples = self.num_samples

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

        for metric in self.metrics.get_compute_order():
            # RootMetric has no dependency per-definition
            # and is computed automatically
            if isinstance(metric, RootMetric):
                continue

            dependencies = {
                dependency: intermediate_stats[get_metric(dependency).name]
                for dependency in metric.dependencies
            }

            values = metric(**dependencies)

            intermediate_stats[metric.name] = values

        reported_metrics = {
            metric.name: intermediate_stats[metric.name]
            for metric in self.metrics.get_insert_order()
        }

        return reported_metrics
