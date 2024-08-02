import typing

import numpy as np
import scipy
import scipy.stats
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation
from bayes_conf_mat.stats import truncated_sample


class GammaAggregator(ExperimentAggregation):
    """Samples from the Gamma-conflated distribution.

    Specifically, the aggregate distribution $\\text{Gamma}(\\tilde{\\alpha}, \\tilde{\\beta})$
    ($\\alpha$ is the shape, $\\beta$ the rate parameter) is estimated as:

    $$\\begin{aligned}
        \\tilde{\\alpha}&=\\left[\sum_{i}^{M}\\alpha_{i}\\right]-(M-1) \\\\
        \\tilde{\\beta}&=\\dfrac{1}{\sum_{i}^{M}\\beta_{i}^{-1}}
    \\end{aligned}$$

    where $M$ is the total number of experiments.

    An optional `shifted: bool` argument exists to dynamically estimate the support for the distribution. Can
    help fit to individual experiments, but likely minimally impacts the aggregate distribution.

    Danger: Assumptions:
        - the individual experiment distributions are gamma distributed

    References: Read more:
        1. [Hill, T. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)
        3. ['Gamma distribution' on Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution)

    """

    name = "gamma"
    full_name = "Gamma conflated experiment aggregator"
    aliases = ["gamma", "gamma_conflation"]

    def __init__(self, rng: np.random.BitGenerator, shifted: bool = False) -> None:
        super().__init__(rng=rng)

        self.shifted = shifted

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Estimate the 'loc' variable, i.e. the minimum of the support
        # Improves fit to individual experiment distributions
        # Minimal impact on the conflated distribution
        if self.shifted:
            loc_estimate = np.min(experiment_samples) - 1e-9
        else:
            loc_estimate = bounds[0]

        # Estimate the shape and rate for each distribution
        alphas = []
        betas = []
        for per_experiment_samples in experiment_samples.T:
            alpha, _, beta = scipy.stats.gamma.fit(
                per_experiment_samples, floc=loc_estimate
            )

            alphas.append(alpha)
            betas.append(beta)

        alphas = np.array(alphas)
        betas = np.array(betas)

        # Estimate the parameters of the conflated distribution
        conflated_alpha = np.sum(alphas) - (num_experiments - 1)
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
