import typing

import numpy as np
import scipy
import scipy.stats
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.experiment_aggregation.utils.truncated_sampling import (
    truncated_sample,
)


class GammaAggregator(ExperimentAggregation):
    name = "gamma"
    full_name = "Gamma conflated experiment aggregator"
    aliases = ["gamma", "gamma_conflation"]

    def __init__(self, rng: np.random.BitGenerator, shifted: bool = False) -> None:
        super().__init__(rng=rng)

        self.shifted = shifted

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_experiments, num_samples = distribution_samples.shape

        # Estimate the 'loc' variable, i.e. the minimum of the support
        # Improves fit to individual experiment distributions
        # Minmal impact on the conflated distribution
        if self.shifted:
            loc_estimate = (
                min(np.min(samples) for samples in distribution_samples) - 1e-12
            )
        else:
            loc_estimate = bounds[0]

        # Estimate the shape and rate for each distribution
        alphas = []
        betas = []
        for samples in distribution_samples:
            alpha, _, beta = scipy.stats.gamma.fit(samples, floc=loc_estimate)

            alphas.append(alpha)
            betas.append(beta)

        alphas = np.array(alphas)
        betas = np.array(betas)

        # Estimate the parameters of the conflated distribution
        conflated_alpha = alphas.sum() - (num_experiments - 1)
        conflated_beta = 1 / np.sum(1 / betas)

        # Redefine the sampling distribution
        conflated_distribution = scipy.stats.gamma(
            a=conflated_alpha, scale=conflated_beta, loc=loc_estimate
        )

        # Sample from the distribution, truncating at the bounds of the metric
        conflated_distribution_samples = truncated_sample(
            sampling_distribution=conflated_distribution,
            bounds=(loc_estimate, bounds[1]),
            rng=self.rng,
            num_samples=num_samples,
        )

        return conflated_distribution_samples
