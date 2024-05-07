import typing

import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.stats import truncated_sample


class FEGaussianAggregator(ExperimentAggregation):
    name = "fe_gaussian"
    full_name = "Fixed-effect Gaussian meta-analytical experiment aggregator"
    aliases = ["fe", "fixed_effect", "fe_gaussian", "gaussian"]

    def __init__(self, rng: np.random.BitGenerator) -> None:
        super().__init__(rng=rng)

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_experiments, num_samples = distribution_samples.shape

        # Estimate the means and variances for each distribution
        means = np.mean(distribution_samples, axis=1)
        variances = np.var(distribution_samples, axis=1, ddof=1)

        # Compute the aggregated mean and variance
        # i.e. the inverse-variance weighted mean
        weights = 1 / variances

        agg_mu = np.sum(weights * means) / np.sum(weights)
        agg_var = 1 / np.sum(weights)

        # Redefine the sampling distribution
        conflated_distribution = scipy.stats.norm(
            loc=agg_mu,
            scale=np.sqrt(agg_var),
        )

        # Sample from the distribution, truncating at the bounds of the metric
        conflated_distribution_samples = truncated_sample(
            sampling_distribution=conflated_distribution,
            bounds=bounds,
            rng=self.rng,
            num_samples=num_samples,
        )

        return conflated_distribution_samples
