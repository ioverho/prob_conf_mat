from dataclasses import dataclass

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.distributions import dirichlet_prior, dirichlet_sample


@dataclass(frozen=True)
class ConfusionMatrixSamples:
    """Simple container for holding raw samples from the confusion matrix posterior."""

    p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"]
    p_pred_given_condition: jtyping.Float[
        np.ndarray, " num_samples num_classes num_classes"
    ]
    norm_confusion_matrix: jtyping.Float[
        np.ndarray, " num_samples num_classes num_classes"
    ]


class BayesianConfusionMatrix:
    """
    Simple wrapper class for holding some paramters for Bayesian estimation of confusion matrices.
    """  # noqa: E501

    def __init__(
        self, confusion_matrix, prior_strategy: str = "laplace", seed: int = 0
    ):
        self.confusion_matrix = confusion_matrix
        self.prior_strategy = prior_strategy

        # Base statistics on the confusion matrix
        self.num_classes = confusion_matrix.shape[0]
        self.num_predictions = confusion_matrix.sum()

        self.pred_counts = confusion_matrix.sum(axis=0)
        self.condition_counts = confusion_matrix.sum(axis=1)
        self.correct_counts = np.diag(confusion_matrix)

        # Generate prior parameters
        self.prior_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes,)
        )
        self.prior_pred_given_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.num_classes, self.num_classes)
        )

        # Generate posterior parameters
        self.posterior_condition_counts = (
            self.prior_condition_counts + self.condition_counts
        )
        self.posterior_pred_given_condtion_counts = (
            self.prior_pred_given_condition_counts + self.confusion_matrix
        )

        # Control RNG
        self.rng = np.random.default_rng(seed=seed)

    def _sample(
        self,
        num_samples: int,
        condition_counts: jtyping.Float[np.ndarray, " num_classes"],
        pred_given_condition_counts: jtyping.Float[
            np.ndarray, " num_classes num_classes"
        ],
    ) -> ConfusionMatrixSamples:
        p_condition = dirichlet_sample(
            rng=self.rng,
            alphas=condition_counts,
            num_samples=num_samples,
        )

        p_pred_given_condition = dirichlet_sample(
            rng=self.rng,
            alphas=pred_given_condition_counts,
            num_samples=num_samples,
        )

        norm_confusion_matrix = p_pred_given_condition * p_condition[:, :, np.newaxis]

        return ConfusionMatrixSamples(
            p_condition=p_condition,
            p_pred_given_condition=p_pred_given_condition,
            norm_confusion_matrix=norm_confusion_matrix,
        )

    def _use_input_as_sample(
        self,
    ):
        """For debug purposes: uses the input confusion matrix as the sample.

        Returns:
            _type_: _description_
        """
        p_condition = (self.condition_counts / self.condition_counts.sum())[
            np.newaxis, :
        ]

        p_pred_given_condition = (
            self.confusion_matrix / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        )[np.newaxis, :, :]

        norm_confusion_matrix = (self.confusion_matrix / self.confusion_matrix.sum())[
            np.newaxis, :, :
        ]

        return ConfusionMatrixSamples(
            p_condition=p_condition,
            p_pred_given_condition=p_pred_given_condition,
            norm_confusion_matrix=norm_confusion_matrix,
        )

    def sample_prior(
        self,
        num_samples: int,
    ):
        """Sample from the prior distribution.

        Args:
            rng (np.random._generator.Generator): _description_
            num_samples (int): _description_

        Returns:
            _type_: _description_
        """
        return self._sample(
            num_samples,
            self.prior_condition_counts,
            self.prior_pred_given_condition_counts,
        )

    def sample_posterior(
        self,
        num_samples: int,
    ):
        """Sample from the posterior distribution.

        Args:
            rng (np.random._generator.Generator): _description_
            num_samples (int): _description_

        Returns:
            _type_: _description_
        """
        return self._sample(
            num_samples,
            self.posterior_condition_counts,
            self.posterior_pred_given_condtion_counts,
        )

    def sample_null_model(
        self,
        num_samples: int,
    ):
        """
        Sample from the null model distribution.
        It uses the class prevalence from the data, but a random confusion matrix.
        Thus, this should model a random classifier on the used dataset, accountingfor class imbalance.

        Args:
            rng (np.random._generator.Generator): _description_
            num_samples (int): _description_

        Returns:
            _type_: _description_
        """  # noqa: E501

        return self._sample(
            num_samples,
            self.posterior_condition_counts,
            self.prior_pred_given_condition_counts,
        )
