import warnings

import numpy as np
import jaxtyping as jtyping

from distributions import dirichlet_prior, dirichlet_sample


class BayesianConfusionMatrix:
    """
    Simple wrapper class for holding some paramters for Bayesian estimation of confusion matrices.
    """  # noqa: E501

    def __init__(self, confusion_matrix, prior_strategy: str = "laplace"):
        self.confusion_matrix = confusion_matrix
        self.prior_strategy = prior_strategy

        # Base statistics on the confusion matrix
        self.support_size = confusion_matrix.shape[0]
        self.sample_size = confusion_matrix.sum()

        self.pred_counts = confusion_matrix.sum(axis=0)
        self.condition_counts = confusion_matrix.sum(axis=1)
        self.correct_counts = np.diag(confusion_matrix)

        # Generate prior parameters
        self.prior_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.support_size,)
        )
        self.prior_pred_given_condition_counts = dirichlet_prior(
            self.prior_strategy, shape=(self.support_size, self.support_size)
        )

        # Generate posterior parameters
        self.posterior_condition_counts = (
            self.prior_condition_counts + self.condition_counts
        )
        self.posterior_pred_given_condtion_counts = (
            self.prior_pred_given_condition_counts + self.confusion_matrix
        )

    def _sample(
        self,
        rng: np.random._generator.Generator,
        num_samples: int,
        condition_counts: jtyping.Float[jtyping.Array, " support"],
        pred_given_condition_counts: jtyping.Float[jtyping.Array, " support support"],
    ):
        p_condition = dirichlet_sample(
            rng=rng,
            alphas=condition_counts,
            num_samples=num_samples,
        )

        p_pred_given_condition = dirichlet_sample(
            rng=rng,
            alphas=pred_given_condition_counts,
            num_samples=num_samples,
        )

        norm_confusion_matrix = p_pred_given_condition * p_condition[:, :, np.newaxis]

        return ConfusionMatrixSamples(
            norm_confusion_matrix, p_condition, p_pred_given_condition
        )

    def _use_input_as_sample(
        self,
    ):
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
            norm_confusion_matrix, p_condition, p_pred_given_condition
        )

    def sample_prior(
        self,
        rng: np.random._generator.Generator,
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
            rng,
            num_samples,
            self.prior_condition_counts,
            self.prior_pred_given_condition_counts,
        )

    def sample_posterior(
        self,
        rng: np.random._generator.Generator,
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
            rng,
            num_samples,
            self.posterior_condition_counts,
            self.posterior_pred_given_condtion_counts,
        )

    def sample_null_model(
        self,
        rng: np.random._generator.Generator,
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
            rng,
            num_samples,
            self.posterior_condition_counts,
            self.prior_pred_given_condition_counts,
        )


class ConfusionMatrixSamples:
    """_summary_"""

    def __init__(
        self,
        norm_confusion_matrix: jtyping.Float[
            jtyping.Array, " num_samples support support"
        ],
        p_condition: jtyping.Float[jtyping.Array, " num_samples support"],
        p_pred_given_condition: jtyping.Float[jtyping.Array, " num_samples support"],
    ):
        self.norm_confusion_matrix = norm_confusion_matrix
        self.p_condition = p_condition
        self.p_pred_given_condition = p_pred_given_condition

        self.support_size = self.norm_confusion_matrix.shape[-1]

        # The other marginal distribution
        self.p_pred = self.norm_confusion_matrix.sum(axis=1)
        
        if (self.p_pred == 0).any():
            warnings.warn("Simulated model neglects class, `p_pred' contains 0.")
        
        self.p_condition_given_pred = (
            self.norm_confusion_matrix / self.p_pred[:, np.newaxis, :]
        )

        self._first_order_metrics()

    def _first_order_metrics(
        self,
    ):
        """
        Computes a battery of first-order statistics using the confusion matrix samples. These metrics are defined directly from the confusion matrix, and serve as intermediate variables for more useful metrics.

        Reasoning about multidimensional negatives is hard: only 1 column/row corresponds to a positive, but all other column/rows correspond to the negatives for a class.
        """  # noqa: E501
        # The mass on the diagonal (TP)
        self.diag_mass = np.diagonal(
            a=self.norm_confusion_matrix,
            axis1=1,
            axis2=2,
        )

        # Positives ============================================================
        # Ratios true positives to condition positives
        # TP / (TP + FN) & FN / (TP + FN)
        self.true_positive_rate = np.diagonal(
            a=self.p_pred_given_condition,
            axis1=1,
            axis2=2,
        )
        self.false_negative_rate = 1 - self.true_positive_rate

        # Ratios true positives to predicted positives
        # TP / (TP + FP) & FP / (TP + FP)
        self.positive_predictive_value = np.diagonal(
            a=self.p_condition_given_pred,
            axis1=1,
            axis2=2,
        )
        self.false_discovery_rate = 1 - self.positive_predictive_value

        # Negatives ============================================================
        # Ratios true negatives to condition negatives
        # FP / (TN + FP) & TN / (TN + FP)
        self.false_positive_rate = (self.p_pred - self.diag_mass) / (
            1 - self.p_condition
        )
        self.true_negative_rate = 1 - self.false_positive_rate

        # Ratios true negatives to predicted negatives
        # FN / (TN + FN) & TN / (TN + FN)
        self.false_omission_rate = (self.p_condition - self.diag_mass) / (
            1 - self.p_pred
        )
        self.negative_predictive_value = 1 - self.false_omission_rate
