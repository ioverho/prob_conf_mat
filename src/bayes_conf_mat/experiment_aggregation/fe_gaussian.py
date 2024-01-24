import typing

import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.experiment_aggregation.utils.truncated_sampling import (
    truncated_sample,
)


class FEGaussianAggregator(ExperimentAggregation):
    name = "fe_gaussian"
    full_name = "Fixed-effect Gaussian meta-analytical experiment aggregator"
    aliases = ["fe", "fixed_effect", "fe_gaussian", "gaussian"]

    def __init__(self, rng: np.random.BitGenerator, num_proc: int = 0) -> None:
        super().__init__(rng=rng, num_proc=num_proc)

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        extrema: typing.Tuple[int],
        rng: np.random.BitGenerator,
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

        # Sample from the distribution, truncating at the extrema of the metric
        conflated_distribution_samples = truncated_sample(
            sampling_distribution=conflated_distribution,
            range=extrema,
            rng=rng,
            num_samples=num_samples,
        )

        return conflated_distribution_samples
