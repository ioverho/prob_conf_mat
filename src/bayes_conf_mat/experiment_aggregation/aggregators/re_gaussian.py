import typing

import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.experiment_aggregation.utils import (
    truncated_sample,
    heterogeneity_DL,
    heterogeneity_PM,
)


class REGaussianAggregator(ExperimentAggregation):
    name = "re_gaussian"
    full_name = "Random-effects Gaussian meta-analytical experiment aggregator"
    aliases = ["re", "random_effect", "re_gaussian"]

    def __init__(
        self,
        rng: np.random.BitGenerator,
        paule_mandel_heterogeneity: bool = True,
        hksj_sampling_distribution: bool = True,
    ) -> None:
        super().__init__(rng=rng)
        self.paule_mandel_heterogeneity = paule_mandel_heterogeneity
        self.hksj_sampling_distribution = hksj_sampling_distribution

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_experiments, num_samples = distribution_samples.shape

        # Estimate the means and variances for each distribution
        means = np.mean(distribution_samples, axis=1)
        variances = np.var(distribution_samples, axis=1, ddof=1)

        # Estimate the between-experiment variance
        tau2 = heterogeneity_DL(means, variances)
        if self.paule_mandel_heterogeneity:
            tau2 = heterogeneity_PM(
                means,
                variances,
                init_tau2=tau2,
                maxiter=100,
                # This is still *very* new, experimental
                # Leave False for now
                use_viechtbauer_correction=False,
            )

        weights = 1 / (variances + tau2)

        agg_variance = 1 / np.sum(weights)
        agg_mean = np.sum(weights * means) / np.sum(weights)

        if self.hksj_sampling_distribution:
            # Uses t-distrbution instead of normal distribution
            # More conservative for small number of studies
            q = np.sum(weights * np.power(means - agg_mean, 2)) / (num_experiments - 1)

            # HKSJ factor with correction
            # Strictly more conservative than FE model
            # hksj_factor = np.sqrt(q)
            hksj_factor = max(1.0, np.sqrt(q))

            aggregated_distribution = scipy.stats.t(
                df=num_experiments - 1,
                loc=agg_mean,
                scale=hksj_factor * np.sqrt(agg_variance),
            )

        else:
            # Uses Gaussian distrbution
            aggregated_distribution = scipy.stats.norm(
                loc=agg_mean,
                scale=np.sqrt(agg_variance),
            )

        aggregated_distribution_samples = truncated_sample(
            sampling_distribution=aggregated_distribution,
            bounds=bounds,
            rng=self.rng,
            num_samples=num_samples,
        )

        return aggregated_distribution_samples
